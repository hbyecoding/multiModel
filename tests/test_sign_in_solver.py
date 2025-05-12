import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sign_in_solver import SignInSolver

def test_merge_intervals():
    solver = SignInSolver()
    
    # 测试空区间列表
    assert solver.merge_intervals([]) == []
    
    # 测试单个区间
    assert solver.merge_intervals([(1, 3)]) == [(1, 3)]
    
    # 测试不重叠的区间
    assert solver.merge_intervals([(1, 2), (4, 5)]) == [(1, 2), (4, 5)]
    
    # 测试重叠的区间
    assert solver.merge_intervals([(1, 3), (2, 4), (5, 7), (6, 8)]) == [(1, 4), (5, 8)]
    
    # 测试完全包含的区间
    assert solver.merge_intervals([(1, 6), (2, 4), (3, 5)]) == [(1, 6)]
    
    # 测试大量重叠区间
    intervals = [(i, i + 2) for i in range(1, 10, 2)]  # 创建多个重叠区间
    assert solver.merge_intervals(intervals) == [(1, 10)]
    
    # 测试起点相同的区间
    assert solver.merge_intervals([(1, 5), (1, 3), (1, 4)]) == [(1, 5)]
    
    print('merge_intervals测试通过')

def test_max_continuous_days():
    solver = SignInSolver()
    
    # 测试空区间列表
    assert solver.max_continuous_days(0, 2, []) == 2
    
    # 测试单个区间
    assert solver.max_continuous_days(1, 2, [(1, 3)]) == 5  # 3天连续 + 2张补签卡
    
    # 测试可以完全填补的间隔
    assert solver.max_continuous_days(2, 1, [(1, 2), (4, 5)]) == 5  # 可以填补中间的1天
    
    # 测试无法完全填补的间隔
    assert solver.max_continuous_days(2, 1, [(1, 2), (5, 6)]) == 3  # 间隔太大，只能扩展单个区间
    
    # 测试多个区间和补签卡
    assert solver.max_continuous_days(3, 2, [(1, 3), (5, 6), (8, 9)]) == 6
    
    # 测试大规模连续区间
    large_intervals = [(i, i + 2) for i in range(1, 20, 3)]  # 创建多个间隔区间
    assert solver.max_continuous_days(7, 4, large_intervals) == 11  # 可以连接多个区间
    
    # 测试理论最大值提前结束
    max_intervals = [(i, i) for i in range(1, 10)]  # 创建多个单点区间
    assert solver.max_continuous_days(9, 5, max_intervals) == 14  # 9个点 + 5张补签卡
    
    # 测试复杂场景
    complex_intervals = [(1, 3), (5, 7), (8, 9), (11, 13), (15, 16)]
    assert solver.max_continuous_days(5, 3, complex_intervals) == 9  # 最优解为连接前三个区间
    
    print('max_continuous_days测试通过')

def main():
    test_merge_intervals()
    test_max_continuous_days()
    print('所有测试用例通过')

if __name__ == '__main__':
    main()