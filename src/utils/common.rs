//! 通用工具函数

/// 简单计时工具
pub fn time_it<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed();
    println!("执行时间: {:?}", duration);
    result
}
