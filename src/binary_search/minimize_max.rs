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
