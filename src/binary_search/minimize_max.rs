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
