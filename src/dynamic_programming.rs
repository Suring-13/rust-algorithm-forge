//! 包含dynamic programming相关算法实现

pub mod bitmask; // 状压
pub mod digit; //数位
pub mod game; // 博弈/概率期望
pub mod interval; // 区间
pub mod intro; //入门题
pub mod knapsack; // 背包问题
pub mod optimized; // 数据结构优化
pub mod partition; // 划分问题
pub mod state_machine; //状态机
pub mod tree; // 树形 

#[cfg(test)]
mod tests;
