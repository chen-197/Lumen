use super::*;

impl Tensor {
    #[cfg(feature = "dev-tools")]
    pub fn dev_has_host_grad(&self) -> bool {
        self.0.borrow().grad.is_some()
    }

    #[cfg(feature = "dev-tools")]
    pub fn dev_has_cuda_f32_grad(&self) -> bool {
        self.device() == Device::Cuda && self.0.borrow().cuda_f32_grad.is_some()
    }

    #[cfg(feature = "dev-tools")]
    pub fn dev_has_cuda_i8_data(&self) -> bool {
        self.device() == Device::Cuda && self.0.borrow().cuda_i8_data.is_some()
    }

    #[cfg(feature = "dev-tools")]
    pub fn dev_has_host_f32_data(&self) -> bool {
        self.0.borrow().has_f32_data
    }
}
