// 3281. 范围内整数的最大得分
pub mod n3281 {
    pub fn max_possible_score(mut start: Vec<i32>, d: i32) -> i32 {
        start.sort_unstable();
        let n = start.len();

        // 二分范围
        let mut left = 0i64;
        let mut right = ((start[n - 1] as i64 + d as i64 - start[0] as i64) / (n - 1) as i64) + 1;

        // 检查能否达到 score
        let check = |score: i64| -> bool {
            let mut x = i64::MIN;
            for &s in &start {
                let s = s as i64;
                x = s.max(x + score);
                if x > s + d as i64 {
                    return false;
                }
            }
            true
        };

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left as i32 - 1
    }
}

// 3620. 恢复网络路径
pub mod n3620 {
    pub fn find_max_path_score(edges: Vec<Vec<i32>>, online: Vec<bool>, k: i64) -> i32 {
        let n = online.len();
        let mut g: Vec<Vec<(usize, i32)>> = vec![vec![]; n];
        let mut max_wt = -1;

        // 构建邻接表，只保留在线节点之间的边
        for edge in edges {
            let x = edge[0] as usize;
            let y = edge[1] as usize;
            let wt = edge[2];

            if online[x] && online[y] {
                g[x].push((y, wt));
                // 记录从起点 0 出发的最大权重
                if x == 0 {
                    max_wt = max_wt.max(wt);
                }
            }
        }

        // 二分查找的检查函数
        let check = |lower: i32| -> bool {
            let mut memo: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();

            fn dfs(
                x: usize,
                g: &Vec<Vec<(usize, i32)>>,
                lower: i32,
                n: usize,
                memo: &mut std::collections::HashMap<usize, i32>,
            ) -> i32 {
                // 到达终点
                if x == n - 1 {
                    return 0;
                }

                // 检查缓存
                if let Some(&val) = memo.get(&x) {
                    return val;
                }

                let mut res = i32::MAX;
                for &(y, wt) in &g[x] {
                    if wt >= lower {
                        let sub = dfs(y, g, lower, n, memo);
                        // 存在有效路径, 即 sub 进入过if判断
                        if sub != i32::MAX {
                            res = res.min(sub + wt);
                        }
                    }
                }

                // 存入缓存
                memo.insert(x, res);
                res
            }

            let total = dfs(0, &g, lower, n, &mut memo);
            total != i32::MAX && total as i64 <= k
        };

        // 二分查找主逻辑
        let mut left = 0;
        let mut right = max_wt + 1;

        while left < right {
            let mid = (left + right) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left - 1
    }
}

// 2517. 礼盒的最大甜蜜度
pub mod n2517 {
    pub fn maximum_tastiness(mut price: Vec<i32>, k: i32) -> i32 {
        price.sort_unstable();

        let check = |d: i32| -> bool {
            let mut cnt = 1;
            let mut pre = price[0]; // 先选一个最小的甜蜜度
            for &p in &price {
                if p - pre >= d {
                    // 可以选
                    cnt += 1;
                    pre = p; // 上一个选的甜蜜度
                }
            }
            cnt >= k
        };

        let mut left = 0;
        let mut right = (price.last().unwrap() - price[0]) / (k - 1) + 1;

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left - 1
    }
}
