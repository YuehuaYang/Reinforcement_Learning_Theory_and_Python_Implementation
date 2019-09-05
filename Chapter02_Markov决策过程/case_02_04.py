# 代码清单2-3  导入‘CliffWalking-v0’环境
import gym
import numpy as np


env = gym.make("CliffWalking-v0")
print("观察空间 = {}".format(env.observation_space))
print("动作空间 = {}".format(env.action_space))
print("状态数量 = {}，动作数量 = {}".format(env.nS, env.nA))
print("地图大小 = {}".format(env.shape))


# 代码清单2-4   函数定义： 运行一回合
def play_once(env, policy):
    """
    运行一回合policy
    :param env:
    :param policy:
    :return:
    """

    total_reward = 0
    state = env.reset() # 初始化状态空间并返回第一个观测
    # loc = np.unravel_index(state,env.shape)  # numpy.unravel_index(索引值，shape),获取一个/组 int类型的索引值在一个多维数组中的位置
    # print("状态 = {}，位置等于 = {}".format(state,loc))
    while True:
        action = np.random.choice(env.nA, p = policy[state])  # numpy.random.choice(a, size=None, replace=True, p=None).
        # 可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回。a是int或一个ndarray，int的话索引范围为np.arange(a)，数组的话从数组中取值
        # size，随机取值的个数； p表示各个参数被取值的概率，没有p的时候，各个值是被等概率取值的
        next_state, reward, done, _ = env.step(action) #env.step(),执行一部操作并返回observation，reward，done，info
        loc = np.unravel_index(state, env.shape)
        # print("状态 = {}，位置 = {}，奖励 = {}".format(state, loc, reward))
        total_reward += reward
        if done:
            break
        state = next_state
    return total_reward


# 代码清单2-5  最优策略与随机策略
actions = np.ones(env.shape, dtype=int)  # 0,向上；1，向右；2，向下；3，向左
actions[-1,:] = 0  # 切片，-1表示最后一行
actions[:,-1] = 2  # 切片，-1表示最后一列
optimal_policy = np.eye(4)[actions.reshape(-1)]  # eye(N, M=None, k=0, dtype=<class 'float'>)，N：Number of rows in the output
# M：Number of columns in the output. If None, defaults to `N`.k : int, optional#对角线的位置，0的时候是正对角线，+1就是对角线向上移，-1就是对角线向下移
# reshape(-1) 中的-1是切片，把数据reshape成1行。等效于reshape(1，1).两列为reshape(-1,2)
random_policy = np.random.uniform(size=(env.nS, env.nA))  #numpy.random.uniform(low,high,size)，从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
    #low: 采样下界，float类型，默认值为0；
    #high: 采样上界，float类型，默认值为1；
    #size: 输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出m*n*k个样本，缺省时输出1个值。
random_policy = random_policy / np.sum(random_policy, axis=1)[:,np.newaxis]
    #numpy.sum(A,axis=None),A:进行加和运算的数组或矩阵；axis==None时加和所有元素，==0时沿纵向加和出一维数组，==1时沿横向加和出一维数组，对于一维数组，axis不能等于1



# 代码清单2-6 用Bellman方程求解状态价值和动作价值
# 很重要，详情见Document文件夹下的“001_求解Bellman期望方程的推算过程.jpg图片”
def evaluate_bellman(env, policy, gamma=1.):
    """
    利用Bellman方程求解状态价值函数和动作函数
    :param env:
    :param policy:
    :param gamma:
    :return: 状态价值，动作价值
    """
    a, b = np.eye(env.nS), np.zeros(env.nS)
    for state in range(env.nS - 1):
        for action in range(env.nA):
            # pi: π(a|s)
            pi = policy[state, action]

            for p, next_state, reward, done in env.P[state][action]: # env.P是一个双层的dictionary,只能是env.P[state][action]可以，不能是env.P[state,action]
                # p-->p(s',r|s,a); r-->reward; s'-->next_state; done-->判断是否为终止状态
                a[state, next_state] -= (pi * gamma * p)
                b[state] += (pi * reward * p)
    v = np.linalg.solve(a,b)  # np.linalg.solve(A,B),是求解AX=B中的X，其中B为一维或二维数组
    q = np.zeros([env.nS, env.nA])
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for p, next_state, reward, done in env.P[state][action]:
                q[state,action] += (p * (reward + gamma * v[next_state]))

    return  v, q


# 执行上述代码
if __name__ == "__main__":
    # 代码清单2-7  评估最优策略
    total_reward = play_once(env,optimal_policy)
    print("最有策略总奖励 = {}".format(total_reward))

    # 代码清单2-8  评估随机策略
    total_reward = play_once(env,random_policy)
    print("随机策略总奖励 = {}".format(total_reward))
