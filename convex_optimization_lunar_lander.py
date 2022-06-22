mport numpy as np
import numpy.linalg as lg
import cvxpy as cp
import math
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt


class constrained_optimizer():
  def __init__(self,A,b,s,lb, ub,k_max, lr, epsilon, n, k2_max):
    self.A = A
    self.b = b
    self.n = n
    self.lb = lb
    self.ub = ub
    self.start_point = s
    self.k_max = k_max
    self.lr = lr
    self.epsilon= epsilon
    self.k2_max = k2_max
    self.standard = 2

  def projection(self, x):     #norm 구간 내에 존재하지 않으면 
    temp_x = np.empty((3,0))
    x = x.reshape(self.n,3)
    for i in x:
      if lg.norm(i) < self.lb or lg.norm(i) > self.ub:
        temp = np.empty(0)
        for j in i :
          temp = np.append(temp,j/lg.norm(i)*self.ub)
        temp_x = np.append(temp_x, temp[:,np.newaxis],axis=1)
      else:
        temp_x = np.append(temp_x,i[:,np.newaxis],axis=1)
    temp_x = np.transpose(temp_x)
    temp_x = temp_x.reshape(self.n*3)
    return temp_x

  def confirm_bds(self, x):
    x = x.reshape(self.n,3)
    for i in x:
      if lg.norm(i) < self.lb or lg.norm(i) > self.ub:
        return False
    return True

  def minimize(self,x):
    grad_temp = 2*np.transpose(self.A)@(self.A@x-self.b)
    temp = x - grad_temp*self.lr

    if self.confirm_bds(temp) == False :  #구간 안에 없으면 projection
      temp = self.projection(temp)


    if lg.norm(self.A@temp-self.b) > lg.norm(self.A@x-self.b) :
      self.lr = self.lr/2
      return False
    else: 
      if lg.norm(temp - self.point) < self.epsilon:
        temp_state = guidance.G@self.point + guidance.Q + guidance.first
        for i in temp_state:
          if abs(i) > self.standard:
            self.lr = 1.2*self.lr
            self.point = temp
            return True
        return 'break'
      else:
        self.lr = 1.2*self.lr
        self.point = temp
        return True

  def gradients(self):
    i = 0
    self.point = self.start_point  
    if self.confirm_bds(self.point) == False:
      self.point = self.projection(self.point)

    while(i < self.k_max): #
      j = 0
      while(j<self.k2_max): #
        min_obj = self.minimize(self.point)
        if min_obj == True:
          break
        elif min_obj == 'break':
          break
        j += 1
      if min_obj == 'break':
        break     
      i += 1

class guidance_moon():
  def __init__(self, x0, xdes, n, T):  #x: 상태변수
    self.A = np.eye(6)
    self.B = np.zeros((6,3))
    self.T = T
    self.n = n
    self.dt = self.T/self.n
    for i in range(3):
      self.A[i,i+3]=self.dt
      self.B[i,i]=self.dt**2/2
      self.B[i+3,i]= self.dt
    self.x_0 = x0
    self.x_des = xdes
    self.x = np.zeros((6,n+1))
    self.x[:,0]=self.x_0
    self.G = np.zeros((6,3*self.n))
    self.P = np.zeros((6,6*self.n))
    self.g = 1.63 #달의 중력가속도
    self.make_matrix()
    self.opt_tr()
    self.lamda2 = 100

  def make_matrix(self):
    temp = 0.5*self.g*self.dt**2
    self.b = [0, 0, temp, 0, 0, self.g*self.dt]
    self.bs = self.b* self.n
    for i in range(self.n):
      self.G[:,3*i:3*(i+1)] = np.linalg.matrix_power(self.A,self.n-(i+1))@self.B
    for i in range(self.n):
      self.P[:,6*i:6*(i+1)]=np.linalg.matrix_power(self.A, n-(i+1))
    self.Q = self.P@self.bs

  def opt_tr(self):
    self.first = np.linalg.matrix_power(self.A, self.n)@self.x_0
    self.u_tilde = sla.lsqr(self.G, self.x_des - self.first-self.Q )[0]
    self.u = self.u_tilde.reshape(self.n,3) #0~n-1
    for i in range(self.n):
      self.x[:,i+1]=self.A.dot(self.x[:,i])+self.B.dot(self.u[i,:])+self.b
      """
    fig = plt.figure(figsize=(14,9))
    ax=fig.gca(projection='3d')
    plt.plot(self.x[0,:],self.x[1,:], self.x[2,:] )
    for i in range(6):
      plt.figure()
      plt.plot(ts, self.x[i,:])
    
    norm_u = []
    for i in self.u:
      norm_u.append(lg.norm(i))
    plt.figure()
    plt.plot(ts[:-1], norm_u)"""


  def lamda_matrix(self, lamda):
    self.G_tilde = np.concatenate((self.G[0:3,:],self.lamda2*self.G[3:6,:]),axis=0)
    self.xqf = self.x_des-self.Q-self.first
    self.xqf_tilde = np.zeros(6)
    self.xqf_tilde[0:3] = self.xqf[0:3]
    self.xqf_tilde[3:6] = self.lamda2*self.xqf[3:6]
    self.A_tilde = np.concatenate((self.G_tilde,np.sqrt(lamda)*np.eye(3*self.n)),axis=0)
    self.b_tilde = np.concatenate((self.xqf_tilde,np.zeros(3*self.n)), axis=0)

  def make_svec(self, us):
    self.x_gd = np.empty((6,0))
    self.x_gd = np.append(self.x_gd, self.x_0[:,np.newaxis],axis=1)
    self.us = us.reshape(self.n, 3)
    self.us = np.transpose(self.us)
    for i in range(self.n):
      temp_x = self.A@self.x_gd[:,i] +self.B@self.us[:,i]+self.b
      self.x_gd = np.append(self.x_gd, temp_x[:,np.newaxis],axis=1)

def plot_tr(*data, label, accs, n):
  fig = plt.figure(figsize=(14,9), dpi = 150)
  ax=fig.gca(projection='3d')
  plt.title('trajectory')
  plt.plot(data[0][0,:],data[0][1,:], -data[0][2,:], label=label[0])
 # plt.plot(data[1][0,:],data[1][1,:], -data[1][2,:], label=label[1])
  plt.plot([x_des[0]], [x_des[1]], [-x_des[2]], '*' ,label='target position')
  ax_n = 3
  for i in range(0,n,2):
    ax.quiver(data[0][0,i],data[0][1,i],-data[0][2,i],ax_n*accs[0][0,i],ax_n*accs[0][1,i],ax_n*accs[0][2,i], length=30, colors='k', normalize=True)
 #
  #for i in range(0,n,3):  
    #ax.quiver(data[1][0,i],data[1][1,i],-data[1][2,i],ax_n*accs[1][0,i],ax_n*accs[1][1,i],ax_n*accs[1][2,i], length=1, colors='g')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()

def plot_state(*data):
  plt.figure(figsize=(18,9))
  ylabel1 =['position']*3+['velocity']*3
  ylabel2 = ['x', 'y', 'z']*2
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.plot(ts, data[0][i,:],'.',color=[0.8,0.1,0.5],label='gradient')
    plt.plot(ts, data[1][i,:],color='#D95319',label='cvxpy')
    plt.text(ts[-1], data[0][i,-1],'%5.2f'%data[0][i,-1], label='final_state', color='black', ha='center', va='bottom')
    plt.xlabel('time')
    plt.ylabel(ylabel1[i]+ylabel2[i])
    plt.legend()
    plt.grid()

def plot_force(*data):
  plt.figure(figsize=(14,5))
  ylabel = ['x', 'y', 'z']
  for i in range(3):
    plt.subplot(1,3,i+1)
    plt.plot(ts[:-1], data[0][i,:], label='gradient')
    plt.plot(ts[:-1], data[1][i,:], label='cvxpy')
    plt.xlabel('time')
    plt.ylabel('force'+ylabel[i])
    plt.legend()
    plt.grid()

T = 30
n = 30  
dt = T/n
x_des = np.array([0,0,0,0,0,0])
x_0 =np.array([100,0,-1500,-10,0,80]) 
ts = np.linspace(0,T,n+1)
u_lb = 0
u_ub = 5.8

guidance = guidance_moon(x_0, x_des, n, T)
guidance.lamda_matrix(1)

optimizer1 = constrained_optimizer(guidance.A_tilde,guidance.b_tilde[:np.newaxis], guidance.u_tilde,0,u_ub,5000,0.01,epsilon=0.00001,n=n,k2_max=100)   #input: 1차원벡터
#start = np.random.randn(1800)
#optimizer1 = constrained_optimizer(guidance.A_tilde,guidance.b_tilde[:np.newaxis], start,0,10,1000,0.01,0.000001,n=n,k2_max=100)   #input: 1차원벡터
optimizer1.gradients()

guidance.make_svec(optimizer1.point)

u = cp.Variable((3,n))
x = cp.Variable((6,n+1))
objective = cp.Minimize(cp.sum_squares(u))
constraints = [x[:,0]== x_0, x[:,-1]==x_des ]
for i in range(n):
  constraints += [x[:,i+1]==guidance.A*x[:,i]+guidance.B*u[:,i]+guidance.b]
  constraints += [cp.sum_squares(u[:,i])<=u_ub**2]
problem = cp.Problem(objective, constraints)
problem.solve(verbose=True)

u_temp = np.array(u.value)
x_temp = np.array(x.value)

plot_tr(guidance.x_gd, guidance.x, label=['gradient descent', 'lsqr'], accs = (guidance.us, guidance.u), n=n )
plot_state(guidance.x_gd, x_temp)
plot_force(guidance.us, u_temp)

#force norm
temp1 = np.empty(0)
for i in np.transpose(guidance.us):
  temp1 = np.append(temp1,lg.norm(i))
temp2 = np.empty(0)
for i in np.transpose(u_temp):
  temp2 = np.append(temp2,lg.norm(i))
plt.figure()
plt.title('norm of force')

plt.plot(ts[:-1], temp2, label='cvxpy')
plt.plot(ts[:-1],temp1,'*',color=[0.8,0.1,0.5],label='gradient')
plt.broken_barh([(u_lb, T)], (u_lb, u_ub), \
                alpha = 0.1, label='Feasible region')
plt.xlabel('time')
plt.ylabel('norm')
plt.legend()
plt.grid()
plt.show()
u_temp = u_temp.reshape(n*3)

print(lg.norm(guidance.u_tilde), lg.norm(optimizer1.point),lg.norm(u_temp))