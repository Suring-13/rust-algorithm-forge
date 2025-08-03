//! 包含sliding window相关算法实现

pub mod fixed_length; // 定长滑动窗口
pub mod grouped_loop; // 分组循环
pub mod three_pointers; // 三指针
pub mod two_pointers; // 双指针（单序列/双序列）
pub mod variable_length; // 不定长滑动窗口

#[cfg(test)]
mod tests;
