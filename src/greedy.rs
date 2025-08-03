//! 包含greedy相关算法实现

pub mod basic; // 基本贪心策略
pub mod construction; // 构造与思维题
pub mod interval; // 区间贪心
pub mod lex_order; // 字典序贪心
pub mod regret; // 反悔贪心

#[cfg(test)]
mod tests;
