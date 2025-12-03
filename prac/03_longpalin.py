class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        best_start, best_len = 0,1
        
        def expand_center(l:int ,r:int):
            while l>=0 and r < n and s[l] == s[r]:
                l-=1
                r+=1
            return l+1, r-l-1 #start, length
        
        if n<2:
            return s
        
        for i in range(n):
            st, ln = expand_center(i,i)
            if ln > best_len:
                best_start, best_len = st, ln
                
            st, ln = expand_center(i,i+1)
            if ln > best_len:
                best_start, best_len = st, ln
                
        return s[best_start: best_start + best_len]
    
if __name__ == "__main__":
    sol = Solution()
    print(sol.longestPalindrome(s = "abcabcbb"))