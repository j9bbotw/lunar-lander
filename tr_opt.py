import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import pandas as pd

# 댐핑 0 & u는 중력장+추력 
# 문제는 u의 최적화가 추력의 최적화를 의미하지는 않는다는 것

def sig_u(x):
  temp=0
  for i in x:
    temp+= math.sqrt(sum(i*i))

  return temp
def conv_np(data):
  temp = []
  for i in data:
    i = i.cpu().detach().numpy()
    temp.append(i)
  return np.array(temp)

def plot_tr(data, label):
  data = conv_np(data)
  fig = plt.figure(figsize=(14,9))
  ax=fig.gca(projection='3d')
  plt.plot(data[:,0],data[:,1], data[:,2], label=label)
  plt.xlabel('x')
  plt.ylabel('y')
  #plt.zlabel('z')
  plt.plot([-rad_m/math.sqrt(3)],[384400*1000+rad_m/math.sqrt(3)],[rad_m/math.sqrt(3)],'*', markersize=10, label='Target position')
  return ax

T = 21600
n = 500#0~1000
dt = T/n
ts = np.linspace(0,T,n+1)
A = np.eye(6)
B = np.zeros((6,3))
x_0 = np.array([0,-6371*1000,0,-11.2*1000,0,0])
rad_m = 1737.4*1000
x_des =np.array([-rad_m/math.sqrt(3),384400*1000+rad_m/math.sqrt(3),rad_m/math.sqrt(3),0,0,0])

for i in range(3):
  A[i,i+3]=dt
  B[i,i]=dt**2/2
  B[i+3,i]= dt

def plot_(*x, label, ylabel,sub, plots=False):
  plt.subplot(3,2,sub)
  temp = int(len(x)/2)
  for i in range(temp):
    plt.plot(x[2*i],x[2*i+1], label=label[i])
  plt.xlabel('time')
  plt.ylabel(ylabel)
  plt.legend()
  plt.grid()


class gravity(nn.Module):
  def __init__(self, device='cuda'):
    super(gravity,self).__init__()
    self.device = device
    self.g1 = 3.986004418*10**14 # 지구 중력 변수(G*M)
    self.g2 = 	4.9048695*10**12
    self.vec_em = torch.tensor([0,384400*1000,0]).to(self.device) # 우선 달 공전 고려 X

  def get_coor(self,r2,a):
    coor = torch.zeros((3)).to(self.device)
    temp = math.sqrt(sum(r2*r2)) 
    for i, j in enumerate(r2,0):
      coor[i] = -j/temp * a

    return coor

  def forward(self,r):
    rm = r-self.vec_em
    ue = self.g1/torch.sum(r*r)
    um = self.g2/torch.sum(rm*rm)
    Ue = self.get_coor(r, ue)
    Um = self.get_coor(rm, um)

    return Ue+Um

class guidance_moon():
  def __init__(self, A, B, x0, xdes, n):  #x: 상태변수
    self.A = A
    self.B = B
    self.x_0 = x0
    self.x_des = xdes
    self.x = np.zeros((6,n+1))
    self.x[:,0]=self.x_0
    self.n = n
    self.G = np.zeros((6,3*n))
    self.grav = gravity()

  def opt_tr(self):
    for i in range(self.n):
      self.G[:,3*i:3*(i+1)] = np.linalg.matrix_power(A,self.n-(i+1))@B
    temp = np.linalg.matrix_power(A, self.n)@self.x_0
    self.u = sla.lsqr(self.G, self.x_des - temp )[0].reshape(self.n,3) #0~n-1
    for i in range(self.n):
      self.x[:,i+1]=self.A.dot(self.x[:,i])+self.B.dot(self.u[i,:])

    return self.u, self.x

  def conv_u(self):
    temp = np.transpose(self.x)[:-1]
    temp3=[]
    for i, j in enumerate(temp,0):
      j = torch.Tensor(j).to('cuda')
      temp2 = np.array(self.grav(j[0:3]).cpu())
      temp3.append(self.u[i]-temp2)
    return np.array(temp3)

guidance = guidance_moon(A,B,x_0,x_des,n)

ul, xl = guidance.opt_tr()
u_true = guidance.conv_u()

print("가속도 크기 합:", sig_u(u_true))

df = pd.DataFrame({'ux':ul[:,0],
                   'uy':ul[:,1],
                   'uz':ul[:,2]})

df2 = pd.DataFrame({'px':xl[0,:],
                   'py':xl[1,:],
                   'pz':xl[2,:],
                   'vx':xl[3,:],
                   'vy':xl[4,:],
                   'vz':xl[5,:]})


fig = plt.figure(figsize=(14,9))
ax=fig.gca(projection='3d')
plt.title("tr_lsqr")
plt.plot(xl[0,:],xl[1,:],xl[2,:])
for i in range(0,n,20):
  ax.quiver(xl[0,i],xl[1,i],xl[2,i],20*ul[i,0],20*ul[i,1],20*ul[i,2], length=5000, colors='r')


class moon_lander(nn.Module):
  def __init__(self, x0, xdes, n, T, u, device="cuda"):
    super(moon_lander,self).__init__()
    self.device = device
    self.xdes = xdes
    self.x0 = x_0
    self.n = n
    self.dt = T/n
    self.u = u
    self.make_weights()
    self.initialize()
    self.layer_g = gravity()
    self.xs =[x_0]
    self.gs=[]
  
  def make_weights(self):
    self.weights =[]
    for i in self.u:
      temp = torch.Tensor(i).to(self.device)
      temp.requires_grad = True
      self.weights.append(temp)

  def initialize(self):
    self.xs = torch.zeros((6,self.n)).to(self.device)
    self.A = torch.eye(6).to(self.device)
    self.B = torch.zeros((6,3)).to(self.device)
    self.C = torch.zeros((6,3)).to(self.device)
    for i in range(6):
      if i <3:
        self.A[i,i+3] = self.dt
        self.B[i,i] = self.dt**2/2 
        self.C[i,i] = self.dt**2/2
      else: 
        self.B[i,i-3] = self.dt
        self.C[i,i-3] = self.dt
  def forward(self,x):
    for i in range(self.n):
      temp = self.layer_g(x[0:3])
      self.gs.append(temp)
      x = torch.mv(self.A, x)+ torch.mv(self.B,self.weights[i])+torch.mv(self.C, temp)
      self.xs.append(x)
    return x


def train():
  running_loss=0
  loss_=[]
  for epoch in range(600):
    model.xs=[x_0]
    model.gs=[]
    optimizer.zero_grad()
    output = model.forward(x_0)
    loss_t = criterion(output, x_des)
    loss_t.backward()
    optimizer.step()
    running_loss += loss_t.item()
    loss_.append(loss_t.item())
    if epoch==599:
      plt.figure()
      plt.title("loss")
      plt.plot(range(1,epoch+2), loss_)
      plt.xlabel("epoch")
      plt.ylabel("loss")
      plot_tr(model.xs, label='tr_training')
  return running_loss/600

torch.manual_seed(777)
x_des=torch.Tensor([-rad_m/math.sqrt(3),384400*1000+rad_m/math.sqrt(3),rad_m/math.sqrt(3),0,0,0]).to('cuda')
T = 21600
n = 500
x_0=torch.Tensor([0,-6371*1000,0,-11.2*1000,0,0]).to('cuda')
model = moon_lander(x_0,x_des, n, T,u_true)
optimizer = torch.optim.Adam(model.weights, weight_decay=1000000000)
criterion = nn.MSELoss().to('cuda') 
loss=[]


for i in range(1):
  temp=train()
  loss.append(temp)


# tesor device: cuda -> cpu
x = conv_np(model.xs)
u = conv_np(model.weights)
g = conv_np(model.gs)

ax = plot_tr(model.xs, label='tr_training')
plt.plot(xl[0,:],xl[1,:],xl[2,:], label='tr_lsqr')
plt.legend()

for i in range(0,n,20):
  ax.quiver(xl[0,i],xl[1,i],xl[2,i],20*ul[i,0],20*ul[i,1],20*ul[i,2], length=5000, colors='r')
  ax.quiver(x[i,0],x[i,1],x[i,2],20*u[i,0],20*u[i,1],20*u[i,2], length=5000, colors='g')
  ax.quiver(x[i,0],x[i,1],x[i,2],20*g[i,0],20*g[i,1],20*g[i,2], length=5000, colors='y')


print("가속도 크기 합(학습 후):",sig_u(u))

print(x.shape)
plt.figure(figsize=(14,9))
plot_(ts,xl[0,:], ts,x[:,0], label=['x position_lsqr','x position_training'], ylabel='x position', sub=1)
plot_(ts,xl[1,:], ts,x[:,1], label=['y position_lsqr','y position_training'], ylabel='y position', sub=2)
plot_(ts,xl[2,:], ts,x[:,2], label=['z position_lsqr','z position_training'], ylabel='z position', sub=3)
plot_(ts,xl[3,:], ts,x[:,3],label=['x velocity_lsqr', 'x_velocity_training'], ylabel='x velocity', sub=4)
plot_(ts,xl[4,:], ts,x[:,4],label=['y velocity_lsqr', 'y_velocity_training'], ylabel='y velocity',sub=5)
plot_(ts,xl[5,:], ts,x[:,5], label=['z velocity_lsqr', 'z_velocity_training'], ylabel='z velocity',sub=6)

plt.figure(figsize=(14,9))
t=ts[:-1]
plot_(t,ul[:,0], t, u_true[:,0],t,u[:,0], label=['x force_lsqr','x force-g_lsqr', 'x force_training'],ylabel='x force', sub=1)
plot_(t,ul[:,1], t, u_true[:,0],t,u[:,1],label=['y force','y force-g_lsqr', 'y force_training'], ylabel='y force',sub=2)
plot_(t,ul[:,2], t, u_true[:,0],t,u[:,2],label=['z force','z force-g_lsqr', 'z force_training'],ylabel='z force',sub=3)
