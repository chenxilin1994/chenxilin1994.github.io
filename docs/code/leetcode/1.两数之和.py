#
# @lc app=leetcode.cn id=1 lang=python3
#
# [1] 两数之和
#
# https://leetcode.cn/problems/two-sum/description/
#
# algorithms
# Easy (54.65%)
# Likes:    19328
# Dislikes: 0
# Total Accepted:    6.1M
# Total Submissions: 11.1M
# Testcase Example:  '[2,7,11,15]\n9'
#
# 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
#
# 你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。
#
# 你可以按任意顺序返回答案。
#
#
#
# 示例 1：
#
#
# 输入：nums = [2,7,11,15], target = 9
# 输出：[0,1]
# 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
#
#
# 示例 2：
#
#
# 输入：nums = [3,2,4], target = 6
# 输出：[1,2]
#
#
# 示例 3：
#
#
# 输入：nums = [3,3], target = 6
# 输出：[0,1]
#
#
#
#
# 提示：
#
#
# 2 <= nums.length <= 10^4
# -10^9 <= nums[i] <= 10^9
# -10^9 <= target <= 10^9
# 只会存在一个有效答案
#
#
#
#
# 进阶：你可以想出一个时间复杂度小于 O(n^2) 的算法吗？
#
#

# @lc code=start
from typing import List


# 以下写法是错误的，如果target=2*num，则会得到错误答案，所以不能一开始就初始化所有的key和value，必须动态创建
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         index = {nums[i]: i for i in range(len(nums))}
#         for num, i in index.items():
#             diff = target - num
#             if diff in index.keys():
#                 return [i, index[diff]]


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        index = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in index:
                return [
                    index[diff],
                    i,
                ]  # 为什么i一定在后面,因为是动态创建的这个字典,所以遍历到第i个值的时候,diff已经存在了,那么diff的索引必定比i小
            index[num] = i


# @lc code=end
# if __name__ == "__main__":
#     s = Solution()
#     nums = [3, 2, 4]
#     target = 6
#     print(s.twoSum(nums, target))
