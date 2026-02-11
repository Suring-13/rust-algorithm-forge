// 410. 分割数组的最大值
pub mod n410 {
    pub fn split_array(nums: Vec<i32>, k: i32) -> i32 {
        let check = |mx: i32| -> bool {
            let mut cnt = 1;
            let mut s = 0;
            for &x in &nums {
                if s + x <= mx {
                    s += x;
                    continue;
                }
                if cnt == k {
                    return true;
                }
                cnt += 1; // 新划分一段
                s = x;
            }
            false
        };

        let mut left = *nums.iter().max().unwrap();
        let mut right = nums.iter().sum::<i32>();
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        right
    }
}

// 2064. 分配给商店的最多商品的最小值
pub mod n2064 {
    pub fn minimized_maximum(n: i32, quantities: Vec<i32>) -> i32 {
        let check = |mx: i32| -> bool {
            let mut cnt = 0;
            for &q in &quantities {
                // 等价于 q/mx 向上取整
                cnt += (q + mx - 1) / mx;
                // 提前终止
                if cnt > n {
                    return true;
                }
            }
            false
        };

        let mut left = 1; // 最小分配量至少为1
        let mut right = *quantities.iter().max().unwrap(); // 右边界取最大值

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        right
    }
}

// 1760. 袋子里最少数目的球
pub mod n1760 {
    pub fn minimum_size(nums: Vec<i32>, max_operations: i32) -> i32 {
        let check = |m: i32| -> bool {
            let mut total = 0i64;
            for &x in &nums {
                total += ((x - 1) / m) as i64;
                if total > max_operations as i64 {
                    // 提前终止，优化性能
                    return true;
                }
            }
            false
        };

        let mut left = 1;
        let mut right = *nums.iter().max().unwrap();
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        right
    }
}

// 1631. 最小体力消耗路径
pub mod n1631 {
    pub fn minimum_effort_path(heights: Vec<Vec<i32>>) -> i32 {
        let m = heights.len();
        let n = heights[0].len();
        let mut left = 0;
        let mut right = 1_000_000;
        let mut ans = 0;
        let dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)];

        while left < right {
            let mid = left + (right - left) / 2;
            let mut q = std::collections::VecDeque::new();
            let mut seen = vec![vec![false; n]; m];
            q.push_back((0, 0));
            seen[0][0] = true;

            while let Some((x, y)) = q.pop_front() {
                for &(dx, dy) in &dirs {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < m as i32 && ny >= 0 && ny < n as i32 {
                        let nx = nx as usize;
                        let ny = ny as usize;
                        if !seen[nx][ny] && (heights[x][y] - heights[nx][ny]).abs() <= mid {
                            q.push_back((nx, ny));
                            seen[nx][ny] = true;
                        }
                    }
                }
            }

            if seen[m - 1][n - 1] {
                ans = mid;
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        ans
    }
}

// 2439. 最小化数组中的最大值
pub mod n2439 {
    pub fn minimize_array_value(nums: Vec<i32>) -> i32 {
        // 核心检查函数：判断当前limit是否满足条件
        let check = |limit: i32| -> bool {
            let mut extra = 0;
            // 反向遍历，将超出部分向左传递
            for &num in nums.iter().skip(1).rev() {
                let new_num = num as i64 + extra;
                extra = 0.max(new_num - limit as i64);
            }
            nums[0] as i64 + extra <= limit as i64
        };

        let mut left = *nums.iter().min().unwrap();
        let mut right = *nums.iter().max().unwrap();
        // 初始答案设为右边界（最大值一定是合法解）
        let mut ans = right;

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                ans = mid;
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        ans
    }
}

// 2560. 打家劫舍 IV
pub mod n2560 {
    pub fn min_capability(nums: Vec<i32>, k: i32) -> i32 {
        let check = |mx: i32| -> bool {
            let (mut f0, mut f1) = (0, 0);
            for &x in nums.iter() {
                if x > mx {
                    f0 = f1;
                } else {
                    let new_f = f1.max(f0 + 1);
                    f0 = f1;
                    f1 = new_f;
                }
            }
            f1 < k
        };

        let (mut left, mut right) = (*nums.iter().min().unwrap(), *nums.iter().max().unwrap() + 1);

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        right
    }
}

// 778. 水位上升的泳池中游泳
pub mod n778 {
    pub fn swim_in_water(grid: Vec<Vec<i32>>) -> i32 {
        let n = grid.len();
        if n == 0 {
            return 0;
        }

        // 检查在水位不超过 mx 时，能否从 (0,0) 到达 (n-1,n-1)
        let check = |mx: i32| -> bool {
            // 初始化访问数组：n x n 的布尔数组，默认 false（未访问）
            let mut visited = vec![vec![false; n]; n];
            dfs(0, 0, &grid, mx, n, &mut visited)
        };

        // 深度优先搜索核心逻辑
        fn dfs(
            i: usize,
            j: usize,
            grid: &Vec<Vec<i32>>,
            mx: i32,
            n: usize,
            visited: &mut Vec<Vec<bool>>,
        ) -> bool {
            // 到达终点
            if i == n - 1 && j == n - 1 {
                return true;
            }
            // 标记当前位置已访问
            visited[i][j] = true;

            // 四个方向：上、右、下、左
            for &(x, y) in &[(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] {
                // 检查坐标是否在合法范围内
                if x < n && y < n {
                    // 检查：水位限制 + 未访问过 + 递归搜索
                    if grid[x][y] <= mx && !visited[x][y] && dfs(x, y, grid, mx, n, visited) {
                        return true;
                    }
                }
            }
            false
        }

        let mut left = grid[0][0].max(grid[n - 1][n - 1]);
        let mut right = (n * n - 1) as i32;

        while left < right {
            let mid = (left + right) / 2;
            if check(mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        right
    }
}

// 2616. 最小化数对的最大差值
pub mod n2616 {
    pub fn minimize_max(mut nums: Vec<i32>, p: i32) -> i32 {
        nums.sort_unstable();
        let n = nums.len();

        let check = |mx: i32| -> bool {
            let mut cnt = 0;
            let mut i = 0;
            while i < n - 1 {
                if nums[i + 1] - nums[i] <= mx {
                    cnt += 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            cnt < p
        };

        let mut left = 0;
        let mut right = nums[n - 1] - nums[0];

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        right
    }
}

// 3419. 图的最大边权的最小值
pub mod n3419 {
    pub fn min_max_weight(n: i32, edges: Vec<Vec<i32>>, _: i32) -> i32 {
        let n = n as usize;
        if edges.len() < n - 1 {
            return -1;
        }

        // 建图：g[y] 保存 (x, w)
        let mut g = vec![vec![]; n];
        for e in &edges {
            let x = e[0] as usize;
            let y = e[1] as usize;
            let w = e[2];
            g[y].push((x, w));
        }

        // 复用 vis 数组，用当前 upper 做标记，避免重复初始化
        let mut vis = vec![-1; n];

        // 二分判定：最大边权不超过 upper 时，能否从 0 遍历所有点
        let mut check = |upper: i32| -> bool {
            fn dfs(x: usize, upper: i32, g: &Vec<Vec<(usize, i32)>>, vis: &mut Vec<i32>) -> usize {
                vis[x] = upper;
                let mut cnt = 1;
                for &(y, w) in &g[x] {
                    if w <= upper && vis[y] != upper {
                        cnt += dfs(y, upper, g, vis);
                    }
                }
                cnt
            }
            dfs(0, upper, &g, &mut vis) == n
        };

        let max_w = edges.iter().map(|e| e[2]).max().unwrap_or(0);
        let mut left = 1;
        let mut right = max_w + 1;
        let mut ans = right;

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                right = mid;
                ans = mid
            } else {
                left = mid + 1;
            }
        }

        if ans > max_w { -1 } else { ans }
    }
}

// 2513. 最小化两个数组中的最大值
pub mod n2513 {
    pub fn minimize_set(d1: i32, d2: i32, unique_cnt1: i32, unique_cnt2: i32) -> i32 {
        // 求最大公约数 gcd
        fn gcd(a: i64, b: i64) -> i64 {
            if b == 0 { a } else { gcd(b, a % b) }
        }

        // 求最小公倍数 lcm
        fn lcm(a: i64, b: i64) -> i64 {
            a / gcd(a, b) * b
        }

        let d1 = d1 as i64;
        let d2 = d2 as i64;
        let u1 = unique_cnt1 as i64;
        let u2 = unique_cnt2 as i64;

        let l = lcm(d1, d2);

        // 二分查找
        let mut left = 1i64;
        let mut right = (u1 + u2) * 2;
        let mut ans = right;

        while left < right {
            let mid = (left + right) / 2;

            let cnt1 = mid / d2 - mid / l;
            let left1 = 0.max(u1 - cnt1);

            let cnt2 = mid / d1 - mid / l;
            let left2 = 0.max(u2 - cnt2);

            let common = mid - mid / d1 - mid / d2 + mid / l;

            if common >= left1 + left2 {
                ans = mid;
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        ans as i32
    }
}
