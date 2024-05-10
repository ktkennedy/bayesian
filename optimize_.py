import sys, os, pathlib, shutil, pickle
from casadi import MX
import casadi as ca
import numpy as np
from dataclasses import dataclass, field, fields

from dynamics import VehicleActuation
import matplotlib.pyplot as plt


@dataclass
class ModelParams():
    n: int = field(default=3) # dimension state space
    d: int = field(default=2) # dimension input space

    N: int = field(default=50)

    Qs: float = field(default=20.0)
    Qey_Values: list = field(default_factory=lambda: [0.9609004019870951])  # Just an example list
    Qey: np.ndarray= field(init=False)
    Qepsi_Values: list = field(default_factory=lambda: [0.9609004019870951])  # Just an example list
    Qepsi: np.ndarray= field(init=False)
    Qv_Values: list = field(default_factory=lambda: [0.9609004019870951])  # Just an example list
    Qv: np.ndarray= field(init=False)
    #Qey: float = field(default=80.0)
    #Qv: float = field(default=80.0)
    #Qepsi: float = field(default=80.0)
   
    R_a: float = field(default=0.01)
    R_delta: float = field(default=0.01)

    # named constraint
    s_max: float = field(default=np.inf)
    s_min: float = field(default=-np.inf)
    ey_max: float = field(default=np.inf)
    ey_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=np.pi/2)
    e_psi_min: float = field(default=-np.pi/2)
    v_max: float = field(default=np.inf)
    v_min: float = field(default=-np.inf)

    u_a_max: float          = field(default = 3.0)
    u_a_min: float          = field(default = -3.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)

    u_a_rate_max: float     = field(default = 2.0)
    u_a_rate_min: float     = field(default = -2.0)
    u_steer_rate_max: float     = field(default = 10)
    u_steer_rate_min: float     = field(default = -10)


    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')
    
   
        
    def __post_init__(self):
        # TODO Temporary fix
        self.Qey = ca.SX(self.Qey_Values)
        self.Qepsi = ca.SX(self.Qepsi_Values)
        self.Qv = ca.SX(self.Qv_Values)
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([self.s_max,
                                  self.ey_max,
                                  self.e_psi_max,
                                  self.v_max])

        self.state_lb = np.array([self.s_min,
                                  self.ey_min,
                                  self.e_psi_min,
                                  self.v_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])

        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min])

        return    

opt_params = ModelParams(
    solver_dir='/home/im/Desktop/IterativeTrackOptimization/solver',
    optlevel=2,
    N= 30,
    
        
    Qs= 0, 
    Qey_Values= [-0.35105800166214096,-0.9588332868132317,-0.8868147421692265,
0.5013771440996495,-0.21456425449780347,-0.8035774316088462,
-0.8956533085941967,-0.49028114071838047,0.798011361429857,
-0.4882437146321321,-0.8049914348851683,-0.6497307046937488,
0.5421812430418795,-0.3434683143895505,-0.5368023837480029,
-0.12395864551245461,0.006024561504452075,0.3691061236710944,
-0.2532865254443255,0.6827309006298794,0.2436888426818531,
-0.9310970510194585,0.016394574610363977,0.22048967835459443,
-0.9245808533308433,-0.6471761223816956,-0.3582313329637459,
-0.6944853345888384,-0.10255969259879105,-0.16768628890602333,
-0.867456162509304,-0.2985312855419453,0.05135050660554197,
-0.8617866198999364,-0.8236676690067006,0.43128204268875603,
0.7475703972557401,-0.12914541616054898,0.6580622833118228,
-0.20097528415318267,0.5195589763274773,0.5775104673196962,
0.6002217471024356,-0.4092554435405449,0.30004633699935335,
0.13396163980576414,-0.9936514783373647,0.2618343240675167,
-0.5665132119309835,0.7086650459405264,-0.7239732446261127,
0.41341020158443675,-0.47055570135587055,0.6988715078204535,
0.6413187350527292,0.19777632358632302,0.0733400105763169,
0.04339627333975926,-0.8787750895547024,-0.661240441907494,
-0.956217129948298,0.8182598584170651,0.8472174934103147,
-0.21473563883064117,0.1430736307934819,0.10920987214489442,
-0.9999756763567094,-0.5248520026547716,-0.6391124293795352,
0.5554943755364459,-0.9044577981703039,-0.4299636128392097,
-0.18273992633574454,-0.87087520111457,0.022626630948501703,
-0.43783416512882467,0.22243794512615156,0.314576506208613,
-0.9252575181937981,0.26224029907016333,-0.899912773464125,
-0.3015679773348463,0.722024575682455,0.26372047308488167,
-0.8398839741032269,0.7225357239921033,0.5741608103980183,
0.5983751021766965,-0.056654106910031476,0.7222370862715977,
-0.2945381990533473,-0.5215326593150431,-0.5286484447005582,
-0.6375231953229297,-0.0806843794937595,-0.1604050878131329,
-0.6785067057104943,-0.21538487510574722,-0.3742922690173054,
-0.06522124821748854,-0.42570922034404757,-0.8759008863884583,
-0.9236760790554885,0.7059815942713332,-0.3329373792188759,
0.14735777222054436,0.0818261794179378,0.23238251244971564,
0.5834125115433835,0.8234834779183964,-0.12194856669772225,
-0.8384513472763622,-0.9958905228957522,-0.8096803952364606,
0.04424105178877458,-0.009566560537597857,-0.12042402466846713,
0.2837024263035577,-0.9407450770790893,0.5640724774146493,
0.9888751827903839,0.4990777075019419,0.7195313363425475,
0.9073417400902573,-0.6046158451653139,0.7169408974194134,
0.0433784144054401,-0.2978718921988861,-0.3651226530773797,
-0.9996483902758584,0.9787312329997648,0.6542323135548234,
0.4676895397705956,-0.19186194275743285,0.18390878729891935,
-0.14895250916656866,0.7987386698795662,-0.2681635853713953,
-0.2171478511659941,0.0003207262776212527,-0.5476891780090438,
0.8554502427187809,-0.9213058003318155,0.9865253488403769,
0.8913376176374934,0.7464536845920264,-0.7765584631148721,
-0.02622920678425733,-0.6922710416840381,-0.11800578987949173,
-0.16686729379747045,0.8680923263121387,0.9377506911873861,
0.05665475897062611,-0.4579158009447768,-0.3555307377600627,
0.8412408696287288,-0.3440347243455688,-0.19004321834797033,
0.8577942270272176,0.6445764788793586,-0.8882024921824614,
0.4006100652508331,-0.35059825885201157,0.19293970502994395,
-0.6698382810989099,0.10956262215362012,0.3950172623632284,
0.2363428952271167,-0.7981684904537261,0.9333389014274562,
0.048691308625264274,0.2734936916670163,0.048943457252283995,
-0.17117389276143724,0.43022409097188063,0.6784057390208604,
0.13675779956485323,0.43902415576093845,0.382684919255609,
0.010562882965367448,0.5360507460902779,0.7889571327029643,
-0.2041134090151513,0.8427987077063925,0.6721216491013045,
-0.13772735889238086,-0.8364560738203,0.155554892396764,
-0.35388166642584507,-0.455239187146244,0.8282492933848338,
-0.5452614690700639,-0.3132319180237366,0.4287946800492728,
0.5015171193552628,-0.16639982270205844,0.9864454461803032,
-0.34962344787721955,0.6553465685661235,-0.4276524470452592,
0.8379953003837595,-0.08586740810832483,-0.428986183743032,
-0.0022059629510911005,-0.010694429794202431,-0.017664787332163145,
0.13999367325908496,-0.38663256922044376,0.5190758675791709,
0.7152761540746513,0.22493078971197944,0.14007577782507674,
0.6542086103670552,0.6834718928638892,0.31820399919948916,
-0.7797564390636429,0.5221790748024553,-0.7128894421944889,
-0.9747796950580541,-0.8908929543311823,0.7586059148052051,
-0.1909298340030503,-0.3488444050757493,-0.781263758571271,
-0.6219243526507325,0.3124021964680246,-0.04983100335420576,
-0.31329695743688024,-0.25163951070365176,0.35734994229605044,
-0.16020989676032582,-0.8593693154306763,0.7909316242567124,
-0.013373756721990926,-0.5236991247948855,-0.21599853626463017,
0.6087129036826726,0.8481397906035386,-0.011859489057705952,
0.6175578613525485,-0.42768819741443265,0.14310602886160506,
-0.21286342116398926,0.19536658221354775,-0.2505885388896403,
-0.39651458350591784,0.6671383442134913,-0.9275308773136048,
-0.8855987028110404,0.9296064978414837,0.951620578831738,
-0.6919984857350854,0.14565855201468247,0.3947450826919645,
-0.11558878351437629,0.6613464186530473,-0.6714334435266707,
-0.3536936813868874,0.7709442489075922,-0.09817212913808682,
-0.48428274904977897,0.8620833626916864,-0.608560657651882,
0.902380203721912,0.12600125843765175,0.2939230785342599,
0.9828550201778292,0.11660818298822084,-0.30074438100643763,
0.24591060755413086,0.7895623853282063,0.324463617308022,
0.18756750955517276,-0.7867687167982007,-0.5568113415278901,
-0.4175340934822631,0.19102301590396742,0.5968201178970365,
0.7412710544531766,0.7717616561096259,-0.17282635175008676,
-0.850355832545316,-0.6144500952951988,-0.6359993659624161,
0.2671970028705337,0.3233509563023995,-0.015265386656403201,
0.8393670252914924,-0.4203431429326354,-0.6188826672138319,
-0.5186042817510506,0.12169339574336413,0.5511734923554432],
    
    # Qv_Values = [0]*99,
    # Qepsi_Values= [0]*99,
    

    R_a= 0,#0.5*(5.85797333e-02+9.05283030e-02+1.07941342e-01),#0.2*(2.62047423e-02+0.17442368+1.81450776e-01+1.45202861e-01+1.12612956e-01),#+0.15681897+0.15454549),#1.09784202e-1,#4.08260789e-02,
    R_delta= 0,#0.5*(4.46676584e+01+5.27098904e+01+2.32279642e+01),#0.2*(2.94466171e+01+0.10137802+4.87248013e+01+3.67516184e+01+3.66712449e+01),#+0.010105976+0.10058888),# 6.16613402e+01,#4.45603470e+01,

    
    v_max= 6.0,
    v_min= 0.5,
    u_a_min= -3,
    u_a_max= 3,
    
    u_steer_max=0.5,
    u_steer_min=-0.5,

    u_a_rate_max=3,
    u_a_rate_min=-3,
    u_steer_rate_max=5,
    u_steer_rate_min=-5
)

class Optimize():
    def __init__(self, dynamics, track, control_params):
        
        '''
        Define Parameters
        '''

        # Track Params
        self.track = track
        self.track_length = track.track_length
        self.N = dynamics.N
        self.dt = dynamics.dt
        curv_info = track.get_curvature_steps(N=self.N)
        self.s_values, self.curv_values = zip(*curv_info)
        # print(self.s_values,self.curv_values)
        # Dynamics Params
        self.dynamics = dynamics
        self.lencar = dynamics.L  

        self.Nx = dynamics.n_q -1# number of states
        self.Nu = dynamics.n_u # number of inputs
        

        # optimization params
        self.control_params = control_params

        self.Qs = ca.SX(control_params.Qs)
        self.Qey = ca.SX(control_params.Qey)
        self.Qv = ca.SX(control_params.Qv)
        self.Qepsi = ca.SX(control_params.Qepsi)

        self.R_a = control_params.R_a
        self.R_delta = control_params.R_delta

        # Input Box Constraints
        self.state_ub = control_params.state_ub
        self.state_lb = control_params.state_lb
        self.state_ub[1] = 0.5
        self.state_lb[1] = -0.5
        self.state_ub[2] = np.pi/2#min(max(self.track.left_bd),max(self.track.right_bd))
        self.state_lb[2] =-np.pi/2#1*min(max(self.track.left_bd),max(self.track.right_bd))
        
        
        self.input_ub = control_params.input_ub
        self.input_lb = control_params.input_lb
        self.input_rate_ub = control_params.input_rate_ub
        self.input_rate_lb = control_params.input_rate_lb

        
        '''
        Define Variables for optimization
        '''

        self.X = ca.SX.sym('X', (self.N + 1)*self.Nx)
        self.U = ca.SX.sym('U', self.N*self.Nu)
        

        self.u_prev = np.zeros(self.Nu)
        self.x_pred = np.zeros((self.N, self.Nx))
        self.u_pred = np.zeros((self.N, self.Nu))
        self.x_ws = None
        self.u_ws = None

        self.cost = 0.0
        self.const = []
    
    # def load_file(self):
    #     self.file_path = '/home/hmcl/ppc_track/optimized_params.txt',   
    #     with open(self.file_path, 'r') as file:
    #         for line in file:
    #             # 쉼표로 분리하여 각 열의 데이터를 추출
    #             values = line.strip().split(',')
    #             if len(values) == 3:  # 데이터가 정확히 3개의 열로 구성된 경우에만 처리
    #                 self.column1.append(float(values[0]))
    #                 self.column2.append(float(values[1]))
    #                 self.column3.append(float(values[2]))
    #     print('column', self.column1)
    #     # 파일에서 읽은 데이터를 Qey_Values, Qv_Values, Qepsi_Values에 할당
    #     self.Qey_Values = self.column1
    #     self.Qv_Values = self.column2
    #     self.Qepsi_Values = self.column3
    #     print(self.Qepsil_Values)
    def solve_optimization(self, state):
        x0, _ = self.dynamics.state2qu(state)
        print(x0)
        ## Cost
        # https://ieeexplore.ieee.org/document/9340640 (7)
        ds = self.track_length / (self.N - 1)
        #print("tracklength,ds", self.track_length,ds)
        v = self.X[2::self.Nx]
        v = (v[1:] + v[:-1]) / 2
        
        #print(self.s_values)
        for i in range(v.shape[0]):
            self.cost += (ds / (v[i] + 1e-12))
        
        for t in range(self.N):
            qey_t = self.Qey[3*t]
            qepsi_t = self.Qey[3*t+1]
            qv_t = self.Qey[3*t+2]
            #qepsi_t = self.Qepsi[t]
            #qv_t = self.V[t]
            self.cost += self.X[self.Nx*t]*qey_t*self.X[self.Nx*t]
            #print(self.X[self.Nx*t].shape)
            
            #self.cost += self.X[self.Nx*t]*self.Qey*self.X[self.Nx*t]
            self.cost += self.X[self.Nx*t+1]*qepsi_t*self.X[self.Nx*t+1]
            self.cost += self.X[self.Nx*t+2]*qv_t*self.X[self.Nx*t+2]
            if t < self.N-1:
                # v_bar = self.X[self.Nx*(t+1)+3] - self.X[self.Nx*t+3] 
                u_bar = self.U[self.Nu*(t+1):self.Nu*(t+2)] - self.U[self.Nu*t:self.Nu*(t+1)]
                # u_bar = self.U[self.Nu*t:self.Nu*(t+1)]

                self.cost += u_bar[0]*self.R_a*u_bar[0]
                self.cost += u_bar[1]*self.R_delta*u_bar[1]
        ## Constraint for dynamics
       
        for t in range(self.N):
            current_s = x0[0] + t*ds
            
            index = min(range(len(self.s_values)), key=lambda i: abs(self.s_values[i]-current_s))
            sym_c_updated = self.curv_values[index]
            
            #print(current_s)
            #print("cur_s", current_s)
            current_x = self.X[self.Nx*t:self.Nx*(t+1)]
            current_u = self.U[self.Nu*t:self.Nu*(t+1)]
            next_x = self.X[self.Nx*(t+1):self.Nx*(t+2)]
            
            # x(n+1) = x(n) + f(x, u)/v * ds
            updated_dynamics = self.dynamics.f_d_rk4(current_x, current_u,self.dt, sym_c_updated)
            self.const = ca.vertcat(self.const, next_x - updated_dynamics) 
            # x(n, 0) = s(n)
            # min_distance = 80
            # max_distance = 80
            # self.const = ca.vertcat(self.const, self.track_length*(t)/self.N- current_s-min_distance)
            # self.const = ca.vertcat(self.const, current_s - self.track_length*(t)/self.N+max_distance)

        # x(N, 0) = s(N)
        # self.const = ca.vertcat(self.const, self.track_length - self.X[self.Nx * self.N])

        ## Constraint for state (upper bound and lower bound)
        
        #self.lbx = list(x0[1:])
        #self.ubx = list(x0[1:]) 
        # for i in range(self.Nx):
        #     self.const = ca.vertcat(self.const, self.X[i] - self.X[self.Nx*(self.N-1)+i])

        ## Define bounds for state and input variables
        #self.lbx = [-1.4, -np.pi/2, 0] + list(self.state_lb[1:]) * self.N + list(self.input_lb) * self.N
        #self.ubx = [1.4, np.pi/2, 8] + list(self.state_ub[1:]) * self.N + list(self.input_ub) * self.N

        self.lbx = [-1.4,-np.pi/2,0]
        self.ubx = [1.4,np.pi/2,5]
        for t in range(self.N):
            
           self.ubx +=  list(self.state_ub[1:])
           self.lbx +=  list(self.state_lb[1:])

        self.ubx +=  list(self.input_ub)*self.N
        self.lbx +=  list(self.input_lb)*self.N
        
        rounds =1 
        
        points_per_round = self.N // rounds
        for i in range(rounds):
            start_index = i * points_per_round * self.Nx
            end_index = (i+1) * points_per_round * self.Nx - self.Nx
            for j in range(self.Nx):
                self.const = ca.vertcat(self.const, self.X[start_index + j] - self.X[end_index + j])
        
        self.lbg_dyanmics = [0]*ca.vertcat(self.const).size()[0]
        self.ubg_dyanmics = [0]*ca.vertcat(self.const).size()[0]
        #self.lbg_dyanmics = [0]*((self.Nx)*self.N)
        #self.ubg_dyanmics = [0]*((self.Nx)*self.N)
        
        
        ######SOLVE
        opts = {"verbose":False,"ipopt.print_level":0,"print_time":0} #, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
        nlp = {'x':ca.vertcat(self.X,self.U), 'f':self.cost, 'g':self.const}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
                
        if self.x_ws is None:
            warnings.warn('Initial guess of open loop state sequence not provided, using zeros')
            self.x_ws = np.zeros((self.N, self.n))
        if self.u_ws is None:
            warnings.warn('Initial guess of open loop input sequence not provided, using zeros')
            self.u_ws = np.zeros((self.N, self.d))

        x_ws = np.zeros((self.N+1, self.Nx))
        x_ws[0] = x0[1:]
        x_ws[1:] = self.x_ws
        x_init = ca.vertcat(np.concatenate(x_ws), np.concatenate(self.u_ws))

        sol = self.solver(x0=x_init, lbx = self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics,ubg=self.ubg_dyanmics)

        x = sol['x']
        print(x)
        g = sol['g']
        print("opt", x[0])
        idx = (self.N+1)*self.Nx
        for i in range(0,self.N):
            self.u_pred[i, 0] = x[idx+self.Nu*i].__float__()
            self.u_pred[i, 1] = x[idx+self.Nu*i + 1].__float__()
       
        for i in range(0,self.N):
            self.x_pred[i, 0] = x[self.Nx*i].__float__()
            self.x_pred[i, 1] = x[self.Nx*i + 1].__float__()
            self.x_pred[i, 2] = x[self.Nx*i + 2].__float__()
            # self.x_pred[i, 3] = x[self.Nx*i + 3].__float__()

        # for i in range(0,self.N-1):
        #     s_next = x[1+self.Nx*(i+1)].__float__()
        #     s_ = x[1+self.Nx*(i)].__float__()
        #     vel = (s_next -s_)/(opt_time/self.N)
        #     print(vel)
        x = np.zeros((len(self.x_pred),1))
        y = np.zeros((len(self.x_pred),1))
        psi = np.zeros((len(self.x_pred),1))
        
        for i in range(0,len(self.x_pred)):
            curv_info = self.track.get_curvature_steps(N=self.N)
            self.s_values, self.curv_values = zip(*curv_info)
            x[i], y[i], psi[i] = self.track.local_to_global(np.array([self.s_values[i], self.x_pred[i,0], self.x_pred[i,1]]))
        
        plt.clf()
        plt.plot(self.track.x,self.track.y,'k')
        plt.plot(self.track.inner[:,0],self.track.inner[:,1],'k')
        plt.plot(self.track.outer[:,0],self.track.outer[:,1],'k')
        plt.plot(x, y,'go')
        plt.show()

        states = np.zeros((len(self.x_pred),3))
        states[:,0] = x.squeeze()
        states[:,1] = y.squeeze()
        states[:,2] = psi.squeeze()

        

        return self.x_pred, self.u_pred, states

    def set_warm_start(self, x_ws, u_ws):
        if x_ws.shape[0] != self.N or x_ws.shape[1] != self.Nx:  # TODO: self.N+1
            raise (RuntimeError(
                'Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    x_ws.shape[0], x_ws.shape[1], self.N, self.Nx)))
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.Nu:
            raise (RuntimeError(
                'Warm start input sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    u_ws.shape[0], u_ws.shape[1], self.N, self.Nu)))

        self.x_ws = x_ws
        self.u_ws = u_ws