import numpy as np

class FuzzyEvaluation:
    def __init__(self, a1=5.5, a2=10.0):
        """
        初始化模糊评价系统
        :param a1: 采矿成本隶属函数的低成本阈值（默认5.5）
        :param a2: 采矿成本隶属函数的高成本阈值（默认10.0）
        """
        self.a1 = a1  # 采矿成本的低成本阈值
        self.a2 = a2  # 采矿成本的高成本阈值
        self.projects = []  # 存储所有项目数据
        self.weights = []  # 存储指标权重
    
    def _mu_A(self, x):
        """可采矿量隶属函数（偏大型）: μ_A(x) = x / 8800"""
        return max(0, min(1, x / 8800))  # 确保结果在[0,1]范围内
    
    def _mu_B(self, x):
        """基建投资隶属函数（偏小型）: μ_B(x) = 1 - x / 8000"""
        return max(0, min(1, 1 - x / 8000))
    
    def _mu_C(self, x):
        """
        采矿成本隶属函数（偏小型分段函数）
        分段规则：
          x ≤ a1 → 1
          a1 < x ≤ a2 → 线性递减
          x > a2 → 0
        """
        if x <= self.a1:
            return 1.0
        elif self.a1 < x <= self.a2:
            return (self.a2 - x) / (self.a2 - self.a1)
        else:
            return 0.0
    
    def _mu_D(self, x):
        """不稳定费用隶属函数（偏小型）: μ_D(x) = 1 - x / 200"""
        return max(0, min(1, 1 - x / 200))
    
    def _mu_E(self, x):
        """净现值隶属函数（偏大型）: μ_E(x) = (x - 50) / 1450"""
        return max(0, min(1, (x - 50) / 1450))
    
    def input_weights(self):
        """
        输入各指标权重（5个指标）
        权重需满足：Σweights = 1, 每个权重∈[0,1]
        """
        print("\n" + "="*50)
        print("步骤1：输入指标权重（5个指标）")
        print("="*50)
        weights = []
        indicators = [
            "可采矿量(μ_A)",
            "基建投资(μ_B)",
            "采矿成本(μ_C)",
            "不稳定费用(μ_D)",
            "净现值(μ_E)"
        ]
        
        for i, indicator in enumerate(indicators):
            while True:
                try:
                    weight = float(input(f"请输入【{indicator}】的权重 (0-1): "))
                    if 0 <= weight <= 1:
                        weights.append(weight)
                        break
                    else:
                        print("权重必须在0-1范围内！")
                except ValueError:
                    print("请输入有效数字！")
        
        # 验证权重总和≈1
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            print(f"警告：权重总和为{total:.2f}，已自动归一化")
            weights = [w/total for w in weights]
        
        self.weights = np.array(weights)
        print(f"最终权重分配: {self.weights.round(4)}")
    
    def input_projects(self):
        """
        输入项目信息：项目数量和各项目指标值
        """
        print("\n" + "="*50)
        print("步骤2：输入项目信息")
        print("="*50)
        n = int(input("请输入评价项目数量: "))
        
        indicators = [
            "可采矿量（吨）",
            "基建投资（万元）",
            "采矿成本（万元/吨）",
            "不稳定费用（万元）",
            "净现值（万元）"
        ]
        
        for i in range(n):
            print(f"\n项目 #{i+1}")
            project_data = []
            
            # 输入各指标值
            for j, indicator in enumerate(indicators):
                while True:
                    try:
                        value = float(input(f"  {indicator}: "))
                        project_data.append(value)
                        break
                    except ValueError:
                        print("请输入有效数字！")
            
            # 计算隶属度
            mu_values = [
                self._mu_A(project_data[0]),
                self._mu_B(project_data[1]),
                self._mu_C(project_data[2]),
                self._mu_D(project_data[3]),
                self._mu_E(project_data[4])
            ]
            
            # 存储项目信息
            self.projects.append({
                'id': i+1,
                'raw_data': project_data,
                'membership': mu_values,
                'comprehensive': None
            })
    
    def evaluate(self):
        """
        执行模糊综合评价计算
        使用加权平均算子：综合得分 = Σ(权重 * 隶属度)
        """
        print("\n" + "="*50)
        print("步骤3：执行模糊综合评价计算")
        print("="*50)
        
        for project in self.projects:
            # 计算综合隶属度
            comp_score = np.dot(self.weights, project['membership'])
            project['comprehensive'] = comp_score
        
        # 按综合得分排序
        self.projects.sort(key=lambda x: x['comprehensive'], reverse=True)
    
    def display_results(self):
        """
        显示评价结果（表格形式）
        """
        print("\n" + "="*50)
        print("最终评价结果（按综合得分降序排列）")
        print("="*50)
        
        # 打印表头
        header = ["项目ID", "可采矿量", "基建投资", "采矿成本", "不稳定费", "净现值", "综合得分"]
        print(f"{'':<5}", end="")
        for h in header:
            print(f"{h:<10}", end="")
        print("\n" + "-"*80)
        
        # 打印每个项目数据
        for project in self.projects:
            print(f"{project['id']:<10}", end="")
            # 原始数据
            for val in project['raw_data']:
                print(f"{val:<10.2f}", end="")
            # 综合得分
            print(f"{project['comprehensive']:<10.4f}")
        
        # 打印最佳项目
        best = self.projects[0]
        print("\n最佳项目：")
        print(f"  项目ID: {best['id']} | 综合得分: {best['comprehensive']:.4f}")
        print("  各指标隶属度:")
        indicators = ["可采矿量(μ_A)", "基建投资(μ_B)", "采矿成本(μ_C)", "不稳定费(μ_D)", "净现值(μ_E)"]
        for i, mu in enumerate(best['membership']):
            print(f"    {indicators[i]}: {mu:.4f}")

# 主程序
if __name__ == "__main__":
    evaluator = FuzzyEvaluation(a1=5.5, a2=10.0)  # 初始化评价系统
    
    evaluator.input_weights()      # 步骤1：输入权重
    evaluator.input_projects()     # 步骤2：输入项目数据
    evaluator.evaluate()           # 步骤3：执行评价计算
    evaluator.display_results()    # 步骤4：显示结果