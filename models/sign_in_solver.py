from typing import List, Tuple

class SignInSolver:
    def __init__(self):
        pass
    
    def merge_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """合并重叠的区间，优化处理大量重叠区间的情况
        
        Args:
            intervals: 区间列表，每个元素为(start, end)元组
            
        Returns:
            合并后的区间列表
        """
        if not intervals:
            return []
        
        # 按区间起点排序，如果起点相同则按终点降序排序
        intervals.sort(key=lambda x: (x[0], -x[1]))
        
        merged = []
        current_start, current_end = intervals[0]
        
        for start, end in intervals[1:]:
            if start <= current_end + 1:
                # 如果当前区间与前一个区间重叠或相邻，更新终点
                current_end = max(current_end, end)
            else:
                # 如果不重叠，保存当前区间并开始新区间
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # 添加最后一个区间
        merged.append((current_start, current_end))
        
        return merged
    
    def max_continuous_days(self, n: int, m: int, intervals: List[Tuple[int, int]]) -> int:
        """计算使用最多m张补签卡后的最长连续签到天数，优化处理大规模数据
        
        Args:
            n: 区间数量
            m: 最多可用的补签卡数量
            intervals: 区间列表，每个元素为(start, end)元组
            
        Returns:
            最长连续签到天数
        """
        if not intervals:
            return m
            
        # 合并重叠区间
        merged = self.merge_intervals(intervals)
        max_continuous = 0
        
        # 使用滑动窗口方法计算最长连续天数
        for i in range(len(merged)):
            # 从当前区间开始尝试扩展
            cards_left = m
            total_days = merged[i][1] - merged[i][0] + 1
            current_end = merged[i][1]
            
            # 向后扩展，尝试连接后续区间
            j = i + 1
            while j < len(merged) and cards_left > 0:
                gap = merged[j][0] - current_end - 1
                if gap <= cards_left:
                    # 可以填补间隔
                    cards_left -= gap
                    total_days += (merged[j][1] - merged[j][0] + 1 + gap)
                    current_end = merged[j][1]
                    j += 1
                else:
                    break
            
            # 使用剩余补签卡扩展区间
            total_days += cards_left
            max_continuous = max(max_continuous, total_days)
            
            # 优化：如果当前连续天数已经达到理论最大值，可以提前结束
            theoretical_max = sum(e - s + 1 for s, e in merged) + m
            if max_continuous >= theoretical_max:
                break
        
        return max_continuous