import numpy as np
import math
import scipy.stats


class MomentsAccountant:

    ## 初始化矩会计，lambda默认最大为32，累积对数矩为none
    def __init__(self, epsilon, delta, noise_multiplier, moment_orders=32):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier

        self.moment_orders = moment_orders
        self.accum_log_moments = None
        self.accum_bgts = 0
        self.__curr_steps = 0
        self.finished = False

    ## 计算指定阶数的矩
    def compute_moment(self, sigma, q, lmbd):
        lmbd_int = int(math.ceil(lmbd))
        if lmbd_int == 0:
            return 1.0

        a_lambda_first_term_exact = 0
        a_lambda_second_term_exact = 0
        for i in range(lmbd_int + 1):
            coef_i = scipy.special.binom(lmbd_int, i) * (q ** i) * (1 - q) ** (lmbd - i)
            s1, s2 = 0, 0
            s1 = coef_i * np.exp((i * i - i) / (2.0 * (sigma ** 2)))
            s2 = coef_i * np.exp((i * i + i) / (2.0 * (sigma ** 2)))
            a_lambda_first_term_exact += s1
            a_lambda_second_term_exact += s2

        a_lambda_exact = ((1.0 - q) * a_lambda_first_term_exact +
                          q * a_lambda_second_term_exact)

        return a_lambda_exact

    ## 计算累积对数矩
    def compute_log_moment(self, sigma, q, steps):
        log_moments = []

        for lmbd in range(self.moment_orders + 1):
            log_moment = 0
            moment = self.compute_moment(sigma, q, lmbd)
            log_moment += np.log(moment) * steps
            log_moments.append((lmbd, log_moment))
        if self.accum_log_moments is None:
            self.accum_log_moments = log_moments
        else:
            new_accum_log_moments = []
            for (lamba, accum_log_moments), (lamba, current_log_moments) in zip(self.accum_log_moments, log_moments):
                new_accum_log_moments.append((lamba, accum_log_moments + current_log_moments))
            self.accum_log_moments = new_accum_log_moments
        # print('accum_log_moments:{}'.format(self.accum_log_moments))
        return self.accum_log_moments

    ## 计算累积隐私损失
    def _compute_eps(self, log_moments, delta):
        min_eps = float("inf")

        for moment_order, log_moment in log_moments:
            if moment_order == 0:
                continue
            if math.isinf(log_moment) or math.isnan(log_moment):
                # print("The %d-th order is inf or Nan\n" % moment_order)
                continue
            min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)

        return min_eps

    def get_privacy_spent(self, sigma, q, steps, target_delta):
        log_moments = self.compute_log_moment(sigma, q, steps)

        return self._compute_eps(log_moments, target_delta)

    def precheck(self, dataset_size, batch_size, loc_steps):
        '''Pre-check if the current client could participate in next round'''

        if self.finished:
            return False

        # Then we need to check if client will exhaust her budget in the following round, i.e., temp_accum_bgts > epsilon.
        tmp_steps = self.__curr_steps + loc_steps
        q = batch_size * 1.0 / dataset_size
        tmp_accum_bgts = self.get_privacy_spent(sigma=self.noise_multiplier, q=q, steps=loc_steps, target_delta=self.delta)
        # tmp_accum_bgts = 10 * q * math.sqrt(tmp_steps * (-math.log10(self.delta))) / self.noise_multiplier

        # If so, set the status as 'finished' and will not participate the rest training anymore; else, return True
        if self.epsilon - tmp_accum_bgts < 0:
            self.finished = True
            return False
        else:
            self.tmp_accum_bgts = tmp_accum_bgts
            return True

    def update(self, loc_steps):
        self.__curr_steps += loc_steps
        self.accum_bgts = self.tmp_accum_bgts
        self.tmp_accum_bgts = 0
        return self.accum_bgts

