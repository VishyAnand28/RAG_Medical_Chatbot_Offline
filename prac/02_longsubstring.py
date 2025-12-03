class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        l = 0
        longest = 0
        sset = set()
        n = len(s)
        # When window is valid we move R, when it's invalid we move L
        # O(n) for range parser
        for r in range(n):
            # remove the seen value from the left
            while s[r] in sset:
                sset.remove(s[l])
                l+=1     
            
            # calculate max window
            w = (r - l) + 1
            longest = max(longest, w)
            sset.add(s[r])
        
        return longest
    
if __name__ == "__main__":
    sol = Solution()
    print(sol.lengthOfLongestSubstring(s = "abcabcbb"))