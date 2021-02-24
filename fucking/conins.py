# 给你 `k` 种面值的硬币，面值分别为 `c1, c2 ... ck`，每种硬币的数量无限，再给一个总金额 `amount`，问你**最少**需要几枚硬币凑出这个金额，如果不可能凑出，算法返回 -1 。
# 算法的函数签名如下：
# // coins 中是可选硬币面值，amount 是目标金额
# int coinChange(int[] coins, int amount);

# 1.base case
def coinChange(coins: list[int], amount: int):
    # 定义：要凑出金额 n，至少要 dp(n) 个硬币
    memo = dict()

    def dp(n):
        if n in memo: return memo[n]
        if n == 0: return 0
        if n < 0: return -1
        # 做选择，选择需要硬币最少的那个结果
        res = float('INF')
        for coin in coins:
            subproblem = dp(n - coin)
            if subproblem == -1: continue
            res = min(res, 1 + subproblem)
        memo[n] = res if res != float('INF') else -1
        return memo[n]

    return dp(amount)


coins = [1, 2, 5, 10, 50, 100]
x = coinChange(coins, 38)
print(x)
