import numpy as np
from bayes_opt import BayesianOptimization
from scipy.optimize import least_squares

class BayesianOptimizer:
    def __init__(self, opt_data, real_data):
        self.opt_data = opt_data
        self.real_data = real_data
        self.pbounds = {}
        self.current_index = 0  # Add this line

        for i in range(len(opt_data)):
            self.pbounds.update({f'c1_{i}': (-0.5, 0.5), f'c2_{i}': (-0.5, 0.5), f'c3_{i}': (-1, 0)})
        
        self.optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=self.pbounds,
            random_state=1,
            verbose=0,
        )
    def objective(self, **kwargs):
        i = self.current_index
        ey = self.opt_data[i, 6]
        epsi = self.opt_data[i, 7]
        v = self.opt_data[i, 8]
        T_opt = self.opt_data[i, 10]
        real_dataset_index = i % len(self.real_data)
        T_real = self.real_data[real_dataset_index][i // len(self.real_data),6]
        model_output = kwargs[f'c1_{i}'] * ey**2 + kwargs[f'c2_{i}'] * epsi**2 + kwargs[f'c3_{i}'] * v**2            
        error = (abs(T_real - T_opt) - model_output)**2
        self.current_index = (self.current_index +1) % len(self.opt_data)  # Add this line
        return -error
    # def objective(self, **kwargs):
        
    #     for i in range(len(self.real_data)):
    #         ey = self.opt_data[i, 6]
    #         epsi = self.opt_data[i, 7]
    #         v = self.opt_data[i, 8]
    #         T_opt = self.opt_data[i, 10]
    #         real_dataset_index = i % len(self.opt_data)
    #         T_real = self.real_data[real_dataset_index][i // len(self.real_data),6]
    #         model_output = kwargs[f'c1_{i}'] * ey**2 + kwargs[f'c2_{i}'] * epsi**2 + kwargs[f'c3_{i}'] * v**2            
    #         print(abs(T_opt - T_real))
    #         error = (abs(T_real - T_opt) - model_output)**2
            
    #     return -error
    # def objective(self, **kwargs):
    #     ey = self.opt_data[:,6]
    #     epsi = self.opt_data[:,7]
    #     v = self.opt_data[:,8]
    #     T_opt = self.opt_data[:,10]
    #     print("T_opt:", T_opt)
    #     T_real = self.real_data[:,6]
    #     print("T_real:", T_real)
    #     model_output = kwargs[f'c1_{i}'] * ey**2 + kwargs[f'c2_{i}']* epsi**2 + kwargs[f'c3_{i}'] * v**2
    #     error = np.sum((T_opt - T_real - model_output) ** 2)
    #     return -error

    def optimize(self):
        self.optimizer.maximize(
            init_points=98,
            n_iter=10,
        )
        best_result = self.optimizer.max
        #print("Best Result:")
        #print("  Params:", best_result['params'])
        #print("  Target:", best_result['target'])
        # for i, res in enumerate(self.optimizer.res):
        #     print(f"Iteration {i}:")
        #     print("  Params:", res['params'])
        #     print("  Target:", res['target'])
        best_params = self.optimizer.max['params']

        # Prepare the lines to write to the file
        c1_params = [f"{value}" for key, value in best_params.items() if key.startswith('c1')]
        c2_params = [f"{value}" for key, value in best_params.items() if key.startswith('c2')]
        c3_params = [f"{value}" for key, value in best_params.items() if key.startswith('c3')]

        # Write the lines to the file
        with open('optimized_params.txt', 'w') as f:
            for c1, c2, c3 in zip(c1_params, c2_params, c3_params):
                f.write(f"{c1},{c2},{c3},\n")

        return best_params

    def transform_data(self):
        ey = self.opt_data[:,6]
        epsi = self.opt_data[:,7]
        v = self.opt_data[:,8]
        T_opt = self.opt_data[:,10]  # Assuming you might need this for some reason

        # Apply the model using the optimized parameters
        model_output = np.zeros_like(ey)
        for i in range(len(ey)):
            model_output[i] = self.optimizer.max['params'][f'c1_{i}'] * ey[i]**2 + self.optimizer.max['params'][f'c2_{i}'] * epsi[i]**2 + self.optimizer.max['params'][f'c3_{i}'] * v[i]**2
        return model_output
    def calculate_gap(self, real_data):
        # Get the transformed data (model output)
        model_output = self.transform_data()

        # Select the relevant column from the real data
        real_data_column = real_data[:, 6] -self.opt_data[:,10]  # Adjust this index if necessary

        # Calculate the gap between the model output and the real data
        gap = real_data_column - model_output

        return gap

# ... (the rest of your code)


    
real_data_list = []
for j in range(5):
    
    real_data = np.loadtxt(f'/home/hmcl/research/ppc_track/result/optimized_traj/optimized_traj{j}.txt', delimiter=',', dtype=float)
    real_data_list.append(real_data) 
    print(f"Shape of real_data {j}: {real_data.shape}")  # 각 real_data의 shape 출력
    print(f"First few rows of real_data {j}:\n{real_data[:5]}")  # 각 real_data의 처음 5행 출력

# real_data = np.loadtxt('/path/to/optimized_traj0.txt', delimiter=',', dtype=float)
opt_data = np.loadtxt('/home/hmcl/Downloads/asd/sdomain/kennedy-main/data/optimized_traj_round1.txt', delimiter=',', dtype=float)
#opt_data = np.loadtxt('/path/to/optimized_traj_round1.txt', delimiter=',', dtype=float)
bayesian_optimizer = BayesianOptimizer(opt_data, real_data_list)
best_params = bayesian_optimizer.optimize()

# Calculate the gap for each real data set
for i, real_data in enumerate(real_data_list):
    gap = bayesian_optimizer.calculate_gap(real_data)
    average_gap = np.mean(gap)
    print(f"Average gap for real data set {i}: {average_gap}")
#transformed_data = bayesian_optimizer.transform_data(best_params)
#print("Best parameters:", best_params)
#print("Transformed data:", transformed_data)

# # 데이터 생성
# N = 100  # 데이터 개수
# data = np.random.rand(N)  # 임의의 데이터
# real_data = np.loadtxt('/home/research/ppc_track/result/optimized_traj0.txt', delimiter=',', dtype=float)
# opt_data = np.loadtxt('/home/Downloads/asd/sdomain/kennedy-main/data/optimized_traj_round1.txt', delimiter=',', dtype=float)
# # 목표 함수 정의: 단순한 선형 조합으로 설정
# def objective(c1, c2, c3):
#     ey = opt_data[:,6]  # 임의의 값
#     epsi = opt_data[:,7]  # 임의의 값
#     v = opt_data[:,8]  # 임의의 값
#     model_output = c1 * ey**2 + c2 * epsi**2 + c3 * v**2
#     # 예제에서는 목표 함수를 최소화하는 것으로 설정, 예를 들어 sum of squares error
#     error = np.sum((data - model_output) ** 2)
#     return -error  # 베이지안 최적화는 최대화를 수행하므로 -error 반환
 
# # 베이지안 최적화 파라미터 범위 설정
# pbounds = {'c1': (-100, 100), 'c2': (-100, 100), 'c3': (-100, 100)}
 
# # 베이지안 최적화 실행
# optimizer = BayesianOptimization(
#     f=objective,
#     pbounds=pbounds,
#     random_state=1,
# )
 
# optimizer.maximize(
#     init_points=5,
#     n_iter=25,
# )
 
# # 최적화 결과 출력
# print("Best parameters:", optimizer.max['params'])
 
# # 최적 계수 사용하여 데이터 변환
# best_params = optimizer.max['params']
# transformed_data = best_params['c1'] * ey + best_params['c2'] * epsi + best_params['c3'] * v
# print("Transformed data:", transformed_data)