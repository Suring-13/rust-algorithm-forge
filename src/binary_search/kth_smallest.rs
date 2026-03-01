// 668. 乘法表中第k小的数
pub mod n668 {
    pub fn find_kth_number(m: i32, n: i32, k: i32) -> i32 {
        // 转换为u32避免负数/溢出（乘法表数值非负，且m/n/k均为正整数）
        let m = m as u32;
        let n = n as u32;
        let k = k as u32;

        // 检查函数：判断≤x的数字数量是否≥k
        let check = |x: u32| -> bool {
            let mut cnt = 0u32;
            // 遍历每一行（1~m）
            for i in 1..=m {
                // 该行≤x的数量：min(x//i, n)
                let row_cnt = n.min(x / i);
                cnt += row_cnt;
                // 提前终止：数量已达标，无需继续遍历
                if cnt >= k {
                    return true;
                }
            }
            cnt >= k
        };

        let mut left = 0u32;
        let mut right = m * n; // 上界为m*n

        while left < right {
            let mid = (left + right) / 2;
            if check(mid) {
                right = mid; // 满足条件，尝试更小的x
            } else {
                left = mid + 1; // 不满足，需要更大的x
            }
        }

        right as i32
    }
}

// 378. 有序矩阵中第 K 小的元素
pub mod n378 {
    pub fn kth_smallest(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let n = matrix.len();

        let check = |mx: i32| -> bool {
            let mut cnt = 0; // matrix 中的 <= mx 的元素个数
            let mut i = 0;
            let mut j = n as i32 - 1; // 从右上角开始
            while i < n && j >= 0 && cnt < k {
                if matrix[i][j as usize] > mx {
                    j -= 1; // 排除第 j 列
                } else {
                    cnt += j + 1; // 从 matrix[i][0] 到 matrix[i][j] 都 <= mx
                    i += 1;
                }
            }
            cnt >= k
        };

        let mut left = matrix[0][0];
        let mut right = matrix[n - 1][n - 1];
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        right
    }
}

// 719. 找出第 K 小的数对距离
pub mod n719 {
    pub fn smallest_distance_pair(mut nums: Vec<i32>, k: i32) -> i32 {
        nums.sort_unstable();
        let k = k as usize;
        let n = nums.len();

        // 检查函数：统计间距≤mx的数对数量是否≥k
        let check = |mx: i32| -> bool {
            let mut cnt = 0usize;
            let mut i = 0usize; // 滑动窗口左指针
            for j in 0..n {
                // 移动左指针，直到nums[j] - nums[i] ≤ mx
                while nums[j] - nums[i] > mx {
                    i += 1;
                }
                // 累加当前右指针j对应的有效数对数量
                cnt += j - i;
                // 提前终止：数量已达标，无需继续遍历
                if cnt >= k {
                    return true;
                }
            }
            cnt >= k
        };

        let mut left = 0;
        let mut right = nums[n - 1] - nums[0];

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                right = mid; // 满足条件，尝试更小的mx
            } else {
                left = mid + 1; // 不满足，需要更大的mx
            }
        }

        right
    }
}

// 878. 第 N 个神奇数字
pub mod n878 {
    pub fn nth_magical_number(n: i32, a: i32, b: i32) -> i32 {
        const MOD: i64 = 1_000_000_007;

        let n = n as i64;
        let a = a as i64;
        let b = b as i64;

        // 最大公约数 gcd
        fn gcd(x: i64, y: i64) -> i64 {
            if y == 0 { x } else { gcd(y, x % y) }
        }
        let lcm = a / gcd(a, b) * b;

        let mut left = a.min(b) + n - 1;
        let mut right = a.min(b) * n;

        while left < right {
            let mid = left + (right - left) / 2;
            let cnt = mid / a + mid / b - mid / lcm;
            if cnt >= n {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        (right % MOD) as i32
    }
}

// 1201. 丑数 III
pub mod n1201 {
    pub fn nth_ugly_number(n: i32, a: i32, b: i32, c: i32) -> i32 {
        let (n, a, b, c) = (n as i64, a as i64, b as i64, c as i64);

        // 最大公约数
        fn gcd(x: i64, y: i64) -> i64 {
            if y == 0 { x } else { gcd(y, x % y) }
        }
        // 最小公倍数
        fn lcm(x: i64, y: i64) -> i64 {
            x / gcd(x, y) * y
        }

        let lcm_ab = lcm(a, b);
        let lcm_ac = lcm(a, c);
        let lcm_bc = lcm(b, c);
        let lcm_abc = lcm(lcm_ab, c);

        // 计算 <=x 的丑数个数（容斥）
        let check = |x: i64| -> bool {
            let count = x / a + x / b + x / c - x / lcm_ab - x / lcm_ac - x / lcm_bc + x / lcm_abc;
            count >= n
        };

        let mut left = a.min(b).min(c) + n - 1;
        let mut right = a.min(b).min(c) * n;

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        right as _
    }
}

// 373. 查找和最小的 K 对数字
pub mod n373 {
    pub fn k_smallest_pairs(nums1: Vec<i32>, nums2: Vec<i32>, k: i32) -> Vec<Vec<i32>> {
        let (m, n) = (nums1.len(), nums2.len());

        let check = |mid: i32| -> bool {
            let mut count = 0;
            let mut i = 0;
            let mut j = n - 1;

            loop {
                if nums1[i] + nums2[j] <= mid {
                    count += j + 1;
                    i += 1;
                    if i == m {
                        break;
                    }
                } else {
                    if j == 0 {
                        break;
                    }
                    j -= 1;
                }
            }
            count >= k as usize
        };

        // 二分查找第 k 小的数对和，思路模仿第378题，假设存在矩阵matrix[i][j] = nums1[i] + nums2[j]
        let (mut left, mut right) = (nums1[0] + nums2[0], nums1[m - 1] + nums2[n - 1] + 1); // 左闭右开 [left, right)
        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        let pair_sum = right;
        let mut ans = Vec::with_capacity(k as usize);

        // 因为存在重复的数对，所以要先找出小于目标值的数对，再找出等于目标值的数对

        // 找数对和小于 pairSum 的数对
        let mut i = n as i32 - 1;
        for &num1 in nums1.iter() {
            while i >= 0 && num1 + nums2[i as usize] >= pair_sum {
                i -= 1;
            }

            for j in 0..i + 1 {
                ans.push(vec![num1, nums2[j as usize]]);
                if ans.len() == k as usize {
                    return ans;
                }
            }
        }

        // 找数对和等于 pairSum 的数对
        i = n as i32 - 1;
        for &num1 in nums1.iter() {
            while i >= 0 && num1 + nums2[i as usize] > pair_sum {
                i -= 1;
            }

            // 反向更快找到足够的数对
            for j in (0..i + 1).rev() {
                if num1 + nums2[j as usize] == pair_sum {
                    ans.push(vec![num1, nums2[j as usize]]);
                }
                if ans.len() == k as usize {
                    return ans;
                }
            }
        }
        ans
    }
}

// 1439. 有序矩阵中的第 k 个最小数组和
pub mod n1439 {
    pub fn kth_smallest(mat: Vec<Vec<i32>>, k: i32) -> i32 {
        let sl: i32 = mat.iter().map(|row| row[0]).sum();
        let sr: i32 = mat.iter().map(|row| row[row.len() - 1]).sum();

        // 判断：和 <= s 的路径数量是否 >= k
        let check = |k: i32, s: i32| -> bool {
            let mut left_k = k;
            dfs(&mat, mat.len() as i32 - 1, s - sl, &mut left_k)
        };

        fn dfs(mat: &Vec<Vec<i32>>, i: i32, s: i32, left_k: &mut i32) -> bool {
            if i < 0 {
                *left_k -= 1;
                return *left_k == 0;
            }

            let i_usize = i as usize;
            let first = mat[i_usize][0];

            for &x in &mat[i_usize] {
                let delta = x - first;
                if delta > s {
                    break;
                }
                if dfs(mat, i - 1, s - delta, left_k) {
                    return true;
                }
            }

            false
        }

        let mut left = sl;
        let mut right = sr + 1;
        let mut ans = 0;

        while left < right {
            let mid = left + (right - left) / 2;
            if check(k, mid) {
                right = mid;
                ans = mid;
            } else {
                left = mid + 1;
            }
        }

        ans
    }
}

// 786. 第 K 个最小的质数分数
pub mod n786 {
    pub fn kth_smallest_prime_fraction(arr: Vec<i32>, k: i32) -> Vec<i32> {
        const EPS: f64 = 1e-8;
        let n = arr.len();
        let mut a = 0;
        let mut b = 0;

        let mut check = |x: f64| -> bool {
            let mut ans = 0;
            let mut i = 0;
            for j in 1..n {
                while i + 1 < n && (arr[i + 1] as f64) / (arr[j] as f64) <= x {
                    i += 1;
                }
                if (arr[i] as f64) / (arr[j] as f64) <= x {
                    ans += (i + 1) as i32;
                }
                if ((arr[i] as f64) / (arr[j] as f64) - x).abs() < EPS {
                    a = arr[i];
                    b = arr[j];
                }
            }
            ans >= k
        };

        let mut left = 0.0;
        let mut right = 1.0;

        while right - left > EPS {
            let mid = left + (right - left) / 2.0;
            if check(mid) {
                right = mid;
            } else {
                left = mid;
            }
        }

        vec![a, b]
    }
}

// 3116. 单面值组合的第 K 小金额
pub mod n3116 {
    pub fn find_kth_smallest(coins: Vec<i32>, k: i32) -> i64 {
        let n = coins.len();
        let k = k as i64;
        if n == 0 {
            return 0;
        }

        // 计算两个数的最大公约数（GCD）
        fn gcd(a: i64, b: i64) -> i64 {
            if b == 0 { a } else { gcd(b, a % b) }
        }

        // 计算两个数的最小公倍数（LCM）
        fn lcm(a: i64, b: i64) -> i64 {
            if a == 0 || b == 0 {
                0
            } else {
                a / gcd(a, b) * b
            }
        }

        // 检查函数：判断 ≤ m 的金额数是否 ≥ k
        let check = |m: i64| -> bool {
            let mut count = 0i64;
            // 枚举所有非空子集（coins无重复，无需去重）
            for mask in 1..(1 << n) {
                let mut lcm_val = 1i64;
                let mut bit_count = 0;
                // 遍历子集的每一位
                for j in 0..n {
                    if (mask >> j) & 1 == 1 {
                        bit_count += 1;
                        lcm_val = lcm(lcm_val, coins[j] as i64);
                        // LCM超过m，提前终止计算
                        if lcm_val > m {
                            break;
                        }
                    }
                }
                // 容斥原理：奇数子集加，偶数子集减
                if lcm_val <= m {
                    if bit_count % 2 == 1 {
                        count += m / lcm_val
                    } else {
                        count -= m / lcm_val
                    };
                }
            }
            count >= k
        };

        let min_coin = *coins.iter().min().unwrap();
        let mut left = min_coin as i64;
        let mut right = min_coin as i64 * k;

        while left < right {
            let mid = left + (right - left) / 2;
            if check(mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        right
    }
}
