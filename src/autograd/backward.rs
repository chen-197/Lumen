use super::*;

impl Tensor {
    /// Borrow the current host gradient, materializing a resident CUDA gradient
    /// on first access. Further borrows share the cached host representation.
    pub fn grad_ref(&self) -> Ref<'_, Option<ArcArray<f32, IxDyn>>> {
        if self.0.borrow().grad.is_none() {
            self.materialize_cuda_grad_to_host();
        }
        let borrow = self.0.borrow();
        Ref::map(borrow, |t| &t.grad)
    }

    #[allow(dead_code)]
    pub(crate) fn has_host_grad(&self) -> bool {
        self.0.borrow().grad.is_some()
    }

    #[inline]
    pub(crate) fn has_pending_grad(&self) -> bool {
        let inner = self.0.borrow();
        inner.grad.is_some() || inner.cuda_f32_grad.is_some()
    }

    // 获取梯度的可变引用
    pub fn grad_mut(&self) -> RefMut<'_, Option<ArcArray<f32, IxDyn>>> {
        if self.0.borrow().grad.is_none() {
            self.materialize_cuda_grad_to_host();
        }
        let mut borrow = self.0.borrow_mut();
        // Mutable host access makes the resident mirror stale, even if a caller
        // replaces or clears the Option. Re-upload lazily when a CUDA consumer asks.
        borrow.cuda_f32_grad = None;
        RefMut::map(borrow, |t| &mut t.grad)
    }

    // 慢路径：返回 owned 的 grad（会拷贝）
    pub fn grad(&self) -> Option<ArrayD<f32>> {
        if self.0.borrow().grad.is_none() {
            self.materialize_cuda_grad_to_host();
        }
        self.0.borrow().grad.as_ref().map(|g| g.to_owned())
    }

    // 快路径：返回共享 grad（clone 仅增 refcount，不复制）
    pub fn grad_arc(&self) -> Option<ArcArray<f32, IxDyn>> {
        if self.0.borrow().grad.is_none() {
            self.materialize_cuda_grad_to_host();
        }
        self.0.borrow().grad.clone()
    }

    #[allow(dead_code)]
    pub(crate) fn cloned_cuda_f32_grad(&self) -> Option<crate::ops::cuda::CudaBuffer> {
        let host = {
            let inner = self.0.borrow();
            if inner.device != Device::Cuda {
                return None;
            }
            if let Some(buffer) = &inner.cuda_f32_grad {
                return Some(buffer.clone());
            }
            let grad = inner.grad.as_ref()?;
            assert_eq!(
                Self::logical_shape(&inner),
                grad.shape(),
                "Gradient shape mismatch"
            );
            if grad.is_empty() {
                return None;
            }
            grad.clone()
        };
        let buffer = cuda_gradient_operation("upload edited gradient", || {
            crate::ops::cuda::upload_f32(&host.iter().copied().collect::<Vec<_>>())
        });
        self.0.borrow_mut().cuda_f32_grad = Some(buffer.clone());
        Some(buffer)
    }

    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.0.borrow().requires_grad
    }

    pub fn zero_grad(&self) {
        let mut inner = self.0.borrow_mut();
        inner.grad = None;
        inner.cuda_f32_grad = None;
    }

    pub fn add_grad(&self, grad: ArrayD<f32>) {
        self.add_grad_with_cuda_buffer(grad, None);
    }

    // Operator VJPs respect requires_grad. Explicit add_grad() and backward
    // seeds retain their existing manual accumulation semantics.
    pub(crate) fn add_backward_grad(&self, grad: ArrayD<f32>) {
        self.accumulate_grad(grad, None, true);
    }

    pub(crate) fn add_backward_grad_with_cuda_buffer(
        &self,
        grad: ArrayD<f32>,
        cuda_grad: Option<crate::ops::cuda::CudaBuffer>,
    ) {
        self.accumulate_grad(grad, cuda_grad, true);
    }

    pub(crate) fn add_grad_with_cuda_buffer(
        &self,
        grad: ArrayD<f32>,
        cuda_grad: Option<crate::ops::cuda::CudaBuffer>,
    ) {
        self.accumulate_grad(grad, cuda_grad, false);
    }

    pub(super) fn accumulate_grad(
        &self,
        grad: ArrayD<f32>,
        cuda_grad: Option<crate::ops::cuda::CudaBuffer>,
        only_if_required: bool,
    ) {
        // Frozen inputs need only a shared borrow: callers may hold read views.
        if only_if_required && !self.requires_grad() {
            return;
        }
        let mut inner = self.0.borrow_mut();
        assert_eq!(
            Self::logical_shape(&inner),
            grad.shape(),
            "Gradient shape mismatch"
        );
        if inner.device != Device::Cuda {
            inner.grad = Some(match &inner.grad {
                Some(old) => (old.to_owned() + &grad).into_shared(),
                None => grad.into_shared(),
            });
            inner.cuda_f32_grad = None;
            return;
        }
        let grad_len = Self::checked_shape_len(Self::logical_shape(&inner), "CUDA gradient");
        if grad_len == 0 {
            if let Some(buffer) = cuda_grad {
                assert_eq!(buffer.len(), 0, "CUDA grad length mismatch");
            }
            // Empty tensors retain a shaped host gradient without a CUDA allocation.
            inner.grad = Some(grad.into_shared());
            inner.cuda_f32_grad = None;
            return;
        }
        if let Some(new_buffer) = cuda_grad {
            assert_eq!(new_buffer.len(), grad_len, "CUDA grad length mismatch");
            if inner.cuda_f32_grad.is_some() {
                // No host total exists when the previous contribution was resident-only.
                // Keep it that way instead of publishing just the new host contribution.
                let host_total = inner
                    .grad
                    .as_ref()
                    .map(|old| (old.to_owned() + &grad).into_shared());
                let total = Self::sum_cuda_grad_buffers(
                    inner
                        .cuda_f32_grad
                        .as_mut()
                        .expect("checked resident gradient"),
                    &new_buffer,
                );
                inner.cuda_f32_grad = Some(total);
                inner.grad = host_total;
                return;
            }
            let (host_total, device_total) = if let Some(old) = &inner.grad {
                let total = old.to_owned() + &grad;
                let buffer = cuda_gradient_operation("upload accumulated gradient", || {
                    crate::ops::cuda::upload_f32(&total.iter().copied().collect::<Vec<_>>())
                });
                (total.into_shared(), buffer)
            } else {
                (grad.into_shared(), new_buffer)
            };
            inner.grad = Some(host_total);
            inner.cuda_f32_grad = Some(device_total);
            return;
        }
        // An explicit host contribution must include any resident-only old total.
        let old_host = inner.grad.clone().or_else(|| {
            inner.cuda_f32_grad.as_ref().map(|buffer| {
                let values = cuda_gradient_operation("download accumulated gradient", || {
                    crate::ops::cuda::download_f32(buffer)
                });
                ArrayD::from_shape_vec(IxDyn(Self::logical_shape(&inner)), values)
                    .expect("CUDA gradient shape mismatch")
                    .into_shared()
            })
        });
        let total = match old_host {
            Some(old) => old.to_owned() + &grad,
            None => grad,
        };
        let buffer = cuda_gradient_operation("upload accumulated gradient", || {
            crate::ops::cuda::upload_f32(&total.iter().copied().collect::<Vec<_>>())
        });
        // Publish both representations only after the upload succeeds.
        inner.grad = Some(total.into_shared());
        inner.cuda_f32_grad = Some(buffer);
    }

    pub(super) fn sum_cuda_grad_buffers(
        old: &mut crate::ops::cuda::CudaBuffer,
        new: &crate::ops::cuda::CudaBuffer,
    ) -> crate::ops::cuda::CudaBuffer {
        assert_eq!(
            old.len(),
            new.len(),
            "CUDA gradient accumulation length mismatch"
        );
        // Shared buffers can also belong to other tensors or upstream seeds.
        // Only a uniquely owned accumulator may be updated in place.
        cuda_gradient_operation("accumulate resident gradient", || {
            if old.has_unique_storage() {
                crate::ops::cuda::add_inplace_f32(old, new)?;
                Ok(old.clone())
            } else {
                crate::ops::cuda::binary_f32_buffer(old, new, crate::ops::cuda::BinaryOp::Add)
            }
        })
    }

    #[allow(dead_code)]
    pub(crate) fn add_cuda_grad_buffer_only(&self, cuda_grad: crate::ops::cuda::CudaBuffer) {
        self.accumulate_cuda_grad_buffer(cuda_grad, false);
    }

    pub(crate) fn add_backward_cuda_grad_buffer_only(
        &self,
        cuda_grad: crate::ops::cuda::CudaBuffer,
    ) {
        self.accumulate_cuda_grad_buffer(cuda_grad, true);
    }

    pub(super) fn accumulate_cuda_grad_buffer(
        &self,
        cuda_grad: crate::ops::cuda::CudaBuffer,
        only_if_required: bool,
    ) {
        if only_if_required && !self.requires_grad() {
            return;
        }
        let mut inner = self.0.borrow_mut();
        assert_eq!(
            inner.device,
            Device::Cuda,
            "add_cuda_grad_buffer_only expects a CUDA tensor"
        );
        let grad_len = Self::checked_shape_len(Self::logical_shape(&inner), "CUDA gradient");
        assert_eq!(cuda_grad.len(), grad_len, "CUDA grad length mismatch");
        if grad_len == 0 {
            inner.grad = Some(ArrayD::zeros(IxDyn(Self::logical_shape(&inner))).into_shared());
            inner.cuda_f32_grad = None;
            return;
        }
        let total = if let Some(old) = inner.cuda_f32_grad.as_mut() {
            Self::sum_cuda_grad_buffers(old, &cuda_grad)
        } else if let Some(host) = &inner.grad {
            let mut old = cuda_gradient_operation("upload previous gradient", || {
                crate::ops::cuda::upload_f32(&host.iter().copied().collect::<Vec<_>>())
            });
            Self::sum_cuda_grad_buffers(&mut old, &cuda_grad)
        } else {
            cuda_grad
        };
        inner.cuda_f32_grad = Some(total);
        inner.grad = None;
    }

    pub(crate) fn materialize_cuda_grad_to_host(&self) {
        let (shape, buffer) = {
            let inner = self.0.borrow();
            let Some(buffer) = inner.cuda_f32_grad.clone() else {
                return;
            };
            (Self::logical_shape(&inner).to_vec(), buffer)
        };

        let host = crate::ops::cuda::download_f32(&buffer)
            .unwrap_or_else(|err| panic!("Failed to download CUDA grad to host: {}", err));
        let grad = Array::from_shape_vec(IxDyn(&shape), host)
            .expect("Failed to materialize host grad from CUDA buffer")
            .into_shared();
        self.0.borrow_mut().grad = Some(grad);
    }

    /// Backpropagate a fresh all-ones upstream gradient.
    ///
    /// Leaf gradients accumulate until `zero_grad()` is called. Intermediate
    /// gradients describe only this traversal, so a retained graph can be reused
    /// without propagating old contributions. Forward values are not recomputed.
    /// Use `backward_with_grad()` for an explicit upstream gradient; existing
    /// values installed with `add_grad()` are not used as backward seeds.
    pub fn backward(&self) {
        self.backward_impl(BackwardSeed::Ones);
    }

    /// Backpropagate an explicit upstream gradient with exactly this tensor's
    /// logical shape. Leaf gradients accumulate just as in `backward()`.
    pub fn backward_with_grad(&self, grad: ArrayD<f32>) {
        assert_eq!(
            self.shape_vec(),
            grad.shape(),
            "backward upstream gradient shape mismatch"
        );
        self.backward_impl(BackwardSeed::Host(grad));
    }

    /// Backpropagate an explicit resident F32 CUDA upstream gradient without
    /// first downloading the seed. Its length must match this tensor exactly.
    /// A shared seed is copied on the device so later caller edits cannot
    /// change accumulated gradients.
    pub fn backward_with_cuda_grad(&self, mut grad: crate::ops::cuda::CudaBuffer) {
        assert_eq!(
            self.device(),
            Device::Cuda,
            "CUDA backward seed requires a CUDA tensor"
        );
        assert_eq!(
            self.len(),
            grad.len(),
            "CUDA backward upstream gradient length mismatch"
        );
        if !grad.is_empty() && !grad.has_unique_storage() {
            grad = cuda_gradient_operation("copy shared CUDA backward seed", || {
                let copy = crate::ops::cuda::alloc_f32(grad.len())?;
                crate::ops::cuda::copy_f32_offset(&copy, 0, &grad, 0, grad.len())?;
                Ok(copy)
            });
        }
        self.backward_impl(BackwardSeed::Cuda(grad));
    }

    pub(super) fn backward_impl(&self, seed: BackwardSeed) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(
            node: &Tensor,
            topo: &mut Vec<Tensor>,
            visited: &mut HashSet<*const TensorData>,
        ) {
            let ptr = node.0.as_ptr() as *const TensorData;
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for parent in &node.0.borrow().parents {
                build_topo(parent, topo, visited);
            }
            topo.push(node.clone());
        }

        build_topo(self, &mut topo, &mut visited);

        // Only leaves are accumulation destinations across backward calls.
        // Device backward closures read the output's resident gradient, so
        // reset both representations before seeding this traversal.
        for node in &topo {
            let is_intermediate = node.0.borrow().backward_op.is_some();
            if is_intermediate {
                node.zero_grad();
            }
        }
        match seed {
            BackwardSeed::Host(grad) => self.add_grad(grad),
            BackwardSeed::Cuda(grad) => self.add_cuda_grad_buffer_only(grad),
            BackwardSeed::Ones => {
                let (shape, seed_cuda_only) = {
                    let inner = self.0.borrow();
                    (
                        Self::logical_shape(&inner).to_vec(),
                        inner.device == Device::Cuda && is_strict_device_execution(),
                    )
                };
                if seed_cuda_only {
                    let len = Self::checked_shape_len(&shape, "CUDA backward seed");
                    if len == 0 {
                        self.add_grad(ArrayD::ones(shape));
                    } else {
                        let buffer = crate::ops::cuda::fill_scalar_f32_buffer(len, 1.0)
                            .unwrap_or_else(|err| {
                                panic!("Failed to create CUDA backward seed: {}", err)
                            });
                        self.add_cuda_grad_buffer_only(buffer);
                    }
                } else {
                    self.add_grad(ArrayD::ones(shape));
                }
            }
        }

        for node in topo.iter().rev() {
            let (has_cuda_only_grad, op_rc, node_device, node_shape, has_host_f32_data) = {
                let inner = node.0.borrow();
                (
                    inner.grad.is_none() && inner.cuda_f32_grad.is_some(),
                    inner.backward_op.clone(),
                    inner.device,
                    Self::logical_shape(&inner).to_vec(),
                    inner.has_f32_data,
                )
            };

            let Some(op) = op_rc else {
                continue;
            };
            let strict_cuda_placeholder = has_cuda_only_grad
                && node_device == Device::Cuda
                && !has_host_f32_data
                && is_strict_device_execution();
            if has_cuda_only_grad && !strict_cuda_placeholder {
                node.materialize_cuda_grad_to_host();
            }

            let grad_arc = if strict_cuda_placeholder {
                Some(ArrayD::<f32>::zeros(IxDyn(&node_shape)).into_shared())
            } else {
                node.0.borrow().grad.clone()
            };
            if let Some(grad) = grad_arc {
                let grad_view = grad.view();
                op(&grad_view.into_dyn());
            }
        }
    }
}
