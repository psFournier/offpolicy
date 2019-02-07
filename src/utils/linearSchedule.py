class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        return self.initial_p + min(float(t) / self.schedule_timesteps, 1) * (self.final_p - self.initial_p)
        # return self.initial_p - T * (self.final_p - self.initial_p)
        # if T == 0:
        #     return 10
        # elif t < self.schedule_timesteps:
        #     return self.initial_p + (float(t) / self.schedule_timesteps) * (self.final_p - self.initial_p)
        # else:
        #     return self.final_p

