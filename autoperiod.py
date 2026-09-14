import numpy as np
import scipy
from astropy.timeseries import LombScargle
from scipy.stats import linregress
from scipy.signal import fftconvolve


class AutoPeriod:
    def __init__(
            self, 
            times, 
            values, 
            fft='lomb_scargle', 
            mc_iterations=1000, 
            confidence_level=.9, 
            release_threshold=1, 
            threshold_method='default',
            acf_method='unbiased',
        ):
        self.times = times - times[0]
        self.values = values
        self.fft = fft
        self.mc_iterations = mc_iterations
        self.confidence_level = confidence_level
        self.release_threshold = release_threshold
        self.threshold_method = threshold_method
        self.acf_method = acf_method
    
    
    def run(self):
        # run lomb-scargle periodogram to get the power spectrum
        self.time_span = self.times[-1] - self.times[0]
        self.time_interval = self.times[1] - self.times[0]
        # centering
        self.values = self.values - np.mean(self.values)

        # spectrum
        self.lomb_scargle = None
        self.freqs, self.powers = self.get_spectrum()
        self.periods = 1 / self.freqs

        # run acf and normalize it
        self.acf = self.autocorrelation()
        if not np.isfinite(self.acf[0]) or np.isclose(self.acf[0], 0.0):
            raise ValueError("ACF lag zero is zero or non-finite")
        # lag zero normalization
        self.acf /= self.acf[0]

        # check the period hints
        self.power_threshold, self.max_powers = self.get_threshold()
        self.power_threshold = self.power_threshold * self.release_threshold

        self.period_hints = self.get_period_hints()
        self.passed_periods = []
        for i, p in self.period_hints:
            valid, period = self.validate_hint(i)
            if valid:
                self.passed_periods.append(period)
        
        self.save_dict = {
            'periods': list(set(self.passed_periods)),
            'period_hints': list(set(self.locate_period_hints())),
            'threshold': self.power_threshold,
        }
        return self.save_dict


    def get_spectrum(self):
        if self.fft == 'lomb_scargle':
            self.lomb_scargle = LombScargle(self.times, self.values)
            self.lomb_scargle_normalization = (
                'psd' if self.threshold_method == 'default' else 'standard'
            )
            self.minimum_frequency = 1 / self.time_span
            self.maximum_frequency = 1 / (self.time_interval * 2)
            freqs, powers = self.lomb_scargle.autopower(
                minimum_frequency=self.minimum_frequency,
                maximum_frequency=self.maximum_frequency,
                normalization=self.lomb_scargle_normalization,
            )
        elif self.fft == 'fft':
            freqs = scipy.fft.fftshift(scipy.fft.fftfreq(len(self.values)))[len(self.values)//2+1:]
            powers = np.abs(scipy.fft.fftshift(scipy.fft.fft(self.values)))[len(self.values)//2+1:] ** 2
        elif self.fft == 'periodogram':
            freqs, powers = scipy.signal.periodogram(self.values)
            freqs = freqs[1:]
            powers = powers[1:]
        else:
            raise ValueError(f"Unknown FFT method: {self.fft}")
        return freqs, powers


    def autocorrelation(self, values=None):
        if values is None:
            values = self.values
        values = np.asarray(values, dtype=float)
        n = values.size
        result = fftconvolve(values, values[::-1], mode='full')
        result = result[result.size // 2:]
        if self.acf_method == 'unbiased':
            # Correct for the number of overlapping pairs at each lag
            result = result / np.arange(n, 0, -1, dtype=float)
        elif self.acf_method == 'biased':
            # Conventional biased autocovariance: use the same denominator n
            result = result / float(n)
        else:  # defensive check for instances created by older code
            raise ValueError("acf_method must be 'biased' or 'unbiased'")
        return result


    def get_threshold(self):
        if self.threshold_method == 'default':
            return self.mc_threshold()
        elif self.threshold_method == 'fap' and self.lomb_scargle is not None:
            return self.fap_threshold()
        elif self.threshold_method == 'baluev' and self.lomb_scargle is not None:
            return self.fap_threshold(fap_method='baluev')
        elif self.threshold_method == 'bootstrap' and self.lomb_scargle is not None:
            return self.fap_threshold(fap_method='bootstrap')
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")

    
    def mc_threshold(self):
        max_powers = []
        shuf = np.copy(self.values)
        for _ in range(self.mc_iterations):
            np.random.shuffle(shuf)
            if self.fft == 'lomb_scargle':
                # Evaluate the shuffle on the same frequency grid
                powers = LombScargle(self.times, shuf).power(
                    self.freqs,
                    normalization=self.lomb_scargle_normalization,
                )
            elif self.fft == 'fft':
                powers = np.abs(scipy.fft.fftshift(scipy.fft.fft(shuf)))[len(shuf)//2+1:] ** 2
            elif self.fft == 'periodogram':
                _, powers = scipy.signal.periodogram(shuf)
                powers = powers[1:]
            else:
                raise ValueError(f"Unknown FFT method: {self.fft}")
            max_powers.append(np.max(powers))
        max_powers.sort()
        return max_powers[int(self.mc_iterations * self.confidence_level)], max_powers


    def fap_threshold(self, fap_method='baluev'):
        if fap_method not in ['baluev', 'bootstrap']:
            raise ValueError(f"Unknown FAP method: {fap_method}")
        threshold = self.lomb_scargle.false_alarm_level(
            1 - self.confidence_level,
            method=fap_method,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
        )
        return threshold, None


    def get_period_hints(self):
        hints = []
        for i in range(len(self.powers)):
            if (self.time_span / 2 > self.periods[i] > self.time_interval * 2) \
                and self.powers[i] > self.power_threshold \
                and (len(self.powers)-2 > i > 2):
                hints.append((i, self.periods[i]))
        hints.sort(key=lambda x: self.powers[x[0]], reverse=True)
        return hints


    def validate_hint(self, p_idx):
        search_min, search_max = self.get_acf_search_range(p_idx)
        min_err = float('inf')
        min_slope1 = 0
        min_slope2 = 0
        for t in range(search_min+1, search_max):
            seg1_x = self.times[search_min:t+1]
            seg1_y = self.acf[search_min:t+1]
            seg2_x = self.times[t:search_max+1]
            seg2_y = self.acf[t:search_max+1]
            slope1, c1, _, _, stderr1 = linregress(seg1_x, seg1_y)
            slope2, c2, _, _, stderr2 = linregress(seg2_x, seg2_y)
            if stderr1 + stderr2 < min_err and seg1_x.size > 2 and seg2_x.size > 2:
                min_err = stderr1 + stderr2
                t_split = t
                min_slope1 = slope1
                min_slope2 = slope2
                min_c1 = c1
                min_c2 = c2
                min_stderr1 = stderr1
                min_stderr2 = stderr2
        angle1 = np.arctan(min_slope1) / (np.pi / 2)
        angle2 = np.arctan(min_slope2) / (np.pi / 2)
        vaild = min_slope1 > min_slope2 and (not np.isclose(np.abs(angle1 - angle2), 0, atol=0.01))
        window = self.acf[search_min:search_max+1]
        peak_idx = np.argmax(window) + search_min
        
        return vaild, self.times[peak_idx].item()


    def get_acf_search_range(self, p_idx):
        # freqs in ascend, periods in descend
        min_period = (self.periods[p_idx] + self.periods[p_idx+1]) / 2 - 1
        max_period = (self.periods[p_idx] + self.periods[p_idx-1]) / 2 + 1
        # operations turn to time axis
        min_idx = np.abs(self.times-min_period).argmin()
        max_idx = np.abs(self.times-max_period).argmin()
        while max_idx - min_idx < 5:
            if min_idx > 0:
                min_idx -= 1
            if max_idx < len(self.times) - 1:
                max_idx += 1
        return min_idx, max_idx


    def locate_period_hints(self):
        located_hints = []
        for i, p in self.period_hints:
            idx = np.abs(self.times - p).argmin()
            located_hints.append(self.times[idx].item())
        return located_hints
