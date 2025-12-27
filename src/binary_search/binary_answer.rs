// 34. 在排序数组中查找元素的第一个和最后一个位置
pub mod n34 {
    pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
        fn custom_partition_point<T, P>(slice: &[T], mut pred: P) -> usize
        where
            P: FnMut(&T) -> bool,
        {
            let mut size = slice.len();
            let (mut left, mut right) = (0, size); // 左闭右开区间 [left, right)
            while left < right {
                // 循环不变量：
                // pred(&slice[left-1]) 为 true
                // pred(&slice[right]) 为 false
                let mid = left + size / 2;
                if pred(&slice[mid]) {
                    left = mid + 1; // 范围缩小到 [mid+1, right)
                } else {
                    right = mid; // 范围缩小到 [left, mid)
                }
                size = right - left;
            }

            // 循环结束后 left = right
            // 此时 pred(&slice[left-1]) 为 true 而 pred(&slice[right])、pred(&slice[right]) 为 false
            // 所以 lefth/right 就是第一个 pred 为 false 的元素下标

            right // 或者 left，写 right 是为了更好记忆。（right满足什么条件，返回的就是第一个满足该条件的值）
        }

        let start = custom_partition_point(&nums, |&x| x < target);
        if start == nums.len() || nums[start] != target {
            return vec![-1, -1];
        }
        let end = custom_partition_point(&nums, |&x| x <= target) - 1;
        vec![start as i32, end as i32]
    }
}
