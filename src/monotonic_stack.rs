//! 包含monotonic stack相关算法实现

pub mod basic; // 基础应用
pub mod contribution; // 贡献法
pub mod lex_order; // 最小字典序
pub mod rectangle; // 矩形问题

#[cfg(test)]
mod tests;
