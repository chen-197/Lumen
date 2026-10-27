use super::*;

#[test]
fn checkpoint_write_reports_flush_failure() {
    struct FlushFailure;
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("injected flush failure"))
        }
    }
    let error = write_checkpoint(FlushFailure, &ModelCheckpoint { params: vec![] }).unwrap_err();
    assert!(error.to_string().contains("injected flush failure"));
}
