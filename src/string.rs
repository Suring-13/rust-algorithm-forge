//! 包含string相关算法实现

pub mod ac_automaton; // AC自动机
pub mod hash; // 字符串哈希
pub mod kmp; // KMP算法
pub mod manacher; // Manacher算法
pub mod subsequence; // 子序列自动机
pub mod suffix_array; // 后缀数组
pub mod z_function; // Z函数

#[cfg(test)]
mod tests;
