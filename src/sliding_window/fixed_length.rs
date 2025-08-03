// 1456. 定长子串中元音的最大数目
pub mod n1456 {
    pub fn max_vowels(s: String, k: i32) -> i32 {
        fn is_vowel(ch: char) -> i32 {
            match ch {
                'a' | 'e' | 'i' | 'o' | 'u' => 1,
                _ => 0,
            }
        }

        let k = k as usize;
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();

        let mut vowel_count: i32 = chars.iter().take(k).map(|&ch| is_vowel(ch)).sum();
        let mut ans = vowel_count;

        for i in k..n {
            vowel_count += is_vowel(chars[i]) - is_vowel(chars[i - k]);
            ans = ans.max(vowel_count);
        }

        ans
    }
}
