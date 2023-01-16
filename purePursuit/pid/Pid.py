class Parameter:
    def __init__(self, Kp, Ki, Kd, Ts, limitMin=None, limitMax=None, Kaw=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.limitMin = limitMin
        self.limitMax = limitMax
        self.Kaw = Kaw

class State:
    def __init__(self, p=0, i=0, d=0, lastInput=0, lastOutput=0, integratorState=0):
        self.p = p
        self.i = i
        self.d = d
        self.lastInput = lastInput
        self.lastOutput = lastOutput
        self.integratorState = integratorState

class Pid:
    def __init__(self, parameter : Parameter, state : State = State()):
        self.parameters = parameter
        self.state = state
        
    def step(self, input_):
        # Calculate proportional
        p = self.parameters.Kp * input_
        
        # Calculate derivative
        d = (input_ - self.state.lastInput) * self.parameters.Kd / self.parameters.Ts;

        # Calculate integral
        i_unscaled = self.state.integratorState + input_ * self.parameters.Ki;
        i = i_unscaled * self.parameters.Ts;

        # Calculate output
        u_desired = p + d + i;
        u = 0
        if self.parameters.limitMin and self.parameters.limitMax:
            u = min(self.parameters.limitMax, max(self.parameters.limitMin, u_desired));
        else:
            u = u_desired
            
        # Anti-windup
        aw_factor = self.parameters.Kaw / (1 + self.parameters.Kaw);
        aw = aw_factor * (u - u_desired);

        # Update stored values (delay elements)
        self.state = State( 
           p=p,
           i=i,
           d=d,
           lastInput=input_,
           lastOutput=u,
           integratorState = i_unscaled + aw
        )

        ### return control output
        return u;
    
    def resetIntegral(self):
        self.state.integratorState = 0.0
        
    def __call__(self, inputValue, dt):
        self.parameters.Ts = dt
        return self.step(inputValue)
        