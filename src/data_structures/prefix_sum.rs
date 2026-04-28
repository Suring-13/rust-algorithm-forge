// 303. 区域和检索 - 数组不可变
pub mod n303 {
    pub struct NumArray {
        pub pre_sum: Vec<i32>,
    }

    impl NumArray {
        pub fn new(nums: Vec<i32>) -> Self {
            let mut pre_sum = vec![0; nums.len() + 1];
            // pre_sum[0] = 0
            // pre_sum[i] = nums[0] + nums[1] + ... + nums[i-1]
            for (i, &num) in nums.iter().enumerate() {
                pre_sum[i + 1] = pre_sum[i] + num;
            }
            Self { pre_sum }
        }

        // [left, right] 区间和
        pub fn sum_range(&self, left: i32, right: i32) -> i32 {
            let l = left as usize;
            let r = right as usize;
            self.pre_sum[r + 1] - self.pre_sum[l]
        }
    }
}

// 3427. 变长子数组求和
pub mod n3427 {
    pub fn subarray_sum(nums: &[i32]) -> i32 {
        let len = nums.len();
        let mut prefix = vec![0i32; len + 1];
        let mut ans = 0i32;

        for (i, &x) in nums.iter().enumerate() {
            prefix[i + 1] = prefix[i] + x;
            let left = i.saturating_sub(x as usize);
            ans += prefix[i + 1] - prefix[left];
        }

        ans
    }
}
