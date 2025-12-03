from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        length_list = len(nums)
        
        # #Brute force
        # for i in range(0, length_list):
        #     for j in range(i+1, length_list):
        #         if nums [i] + nums[j] == target:
        #             return [i,j]
        # return False
        
        seen = {} # store value & index, x is current number
        
        for i, x in enumerate(nums):
            comp = target - x
            if comp in seen:
                return [seen[comp],i]
            #store after checking
            seen[x]=i
                
if __name__ == "__main__":
    sol = Solution()
    print(sol.twoSum([2,7,11,15], target=17))