import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards,\
     batch_dones,batch_cImage, batch_next_cImage = [], [], [], [], [], [], []
    for i in ind:
      state, next_state, action, reward, done = self.storage[i]      
      batch_states.append(np.array(state[1:], copy=False))
      batch_next_states.append(np.array(next_state[1:], copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
      batch_cImage.append(np.array(state[0], copy=False))
      batch_next_cImage.append(np.array(next_state[0], copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), \
        np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1),\
            np.array(batch_cImage), np.array(batch_next_cImage)



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()     

        self.encoder = torch.nn.ModuleList([ 
            torch.nn.Conv2d(1, 8, 3, 1, padding = 0, bias=False), 
            torch.nn.BatchNorm2d(8),            
            nn.Dropout2d(0.1),

            torch.nn.Conv2d(8, 10, 3, 1, padding = 0, bias=False),    
            torch.nn.ReLU(),       
            torch.nn.BatchNorm2d(10),            
            nn.Dropout2d(0.1),
            
            torch.nn.Conv2d(10, 12, 3, 2, padding = 0, bias=False),  
            torch.nn.ReLU(),         
            torch.nn.BatchNorm2d(12),            
            nn.Dropout2d(0.1),
            
            torch.nn.Conv2d(12, 16, 3, 1, padding = 0, bias=False),  
            torch.nn.AdaptiveAvgPool2d((1,1)), 
            
        ])
        self.layer_1 = nn.Linear(state_dim + 15, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x,state):
        for layer in self.encoder:
            x = layer(x)        
        x = x.view(-1, 16)        
        x = torch.cat([x, state], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.encoder_1 = torch.nn.ModuleList([ 
            torch.nn.Conv2d(1, 8, 3, 1, padding = 0, bias=False), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),            
            nn.Dropout2d(0.1),

            torch.nn.Conv2d(8, 10, 3, 1, padding = 0, bias=False), 
            torch.nn.BatchNorm2d(10),            
            nn.Dropout2d(0.1),

            torch.nn.Conv2d(10, 12, 3, 2, padding = 0, bias=False),  
            torch.nn.ReLU(),        
            torch.nn.BatchNorm2d(12),            
            nn.Dropout2d(0.1),          

            torch.nn.Conv2d(12, 16, 3, 1, padding = 0, bias=False), 
            torch.nn.AdaptiveAvgPool2d((1,1)), 
            
        ])        
        self.layer_a1 = nn.Linear(state_dim + 15 + action_dim, 512)
        self.layer_a2 = nn.Linear(512, 256)
        self.layer_a3 = nn.Linear(256, 1)

        self.encoder_2 = torch.nn.ModuleList([ 
            torch.nn.Conv2d(1, 8, 3, 1, padding = 0, bias=False),  
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),            
            nn.Dropout2d(0.1),

            torch.nn.Conv2d(8, 10, 3, 1, padding = 0, bias=False),  
            torch.nn.ReLU(),      
            torch.nn.BatchNorm2d(10),            
            nn.Dropout2d(0.1),

            torch.nn.Conv2d(10, 12, 3, 2, padding = 0, bias=False), 
            torch.nn.ReLU(),        
            torch.nn.BatchNorm2d(12),            
            nn.Dropout2d(0.1),          

            torch.nn.Conv2d(12, 16, 3, 1, padding = 0, bias=False),  
            torch.nn.AdaptiveAvgPool2d((1,1)), 
             
        ])  

        self.layer_b1 = nn.Linear(state_dim + 15 + action_dim, 512)
        self.layer_b2 = nn.Linear(512, 256)
        self.layer_b3 = nn.Linear(256, 1)

    def forward(self, x, state, u):

        x1 = x 
        for layer in self.encoder_1:
            x1 = layer(x1)
        x1 = x1.view(-1, 16)
        x1u = torch.cat([x1, state, u], 1)        
        x1u = F.relu(self.layer_a1(x1u))
        x1u = F.relu(self.layer_a2(x1u))
        x1u = self.layer_a3(x1u)

        x2 = x 
        for layer in self.encoder_2:
            x2 = layer(x2)
        x2 = x2.view(-1, 16)
        x2u = torch.cat([x2, state, u], 1)
        x2u = F.relu(self.layer_b1(x2u))
        x2u = F.relu(self.layer_b2(x2u))
        x2u = self.layer_b3(x2u)
        return x1u, x2u

    def Q1(self, x1, state, u):

        for layer in self.encoder_1:
            x1 = layer(x1)

        x1 = x1.view(-1, 16)
        x1u = torch.cat([x1, state, u], 1)
        x1u = F.relu(self.layer_a1(x1u))
        x1u = F.relu(self.layer_a2(x1u))
        x1u = self.layer_a3(x1u)
        return x1u


# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())    
    self.max_action = max_action

  def select_action(self, state):
    # passing image and other variables of state seperately
    croppedImage = np.expand_dims(state[0],1)
    croppedImage = torch.Tensor(croppedImage).to(device)
    state = np.array(state[1:],dtype=np.float)
    state = np.expand_dims(state,0)
    state = torch.Tensor(state.reshape(1, -1)).to(device)    
    return self.actor(croppedImage,state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r, i, i') from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, \
          batch_dones,batch_image,\
           batch_next_image  = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      image = torch.Tensor(batch_image).to(device)
      next_image = torch.Tensor(batch_next_image).to(device)
      
      # Step 5: From the next state i',s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_image,next_state)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it 
      # in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (i', s’, a’) as input 
      # and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_image,next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, 
      # which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (i, s, a) as 
      # input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(image, state, action)
      
      # Step 11: We compute the loss coming from the two
      #  Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the 
      # parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model
      #  by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(image, state, self.actor(image, state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target 
        # by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target
        #  by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename),map_location = "cpu"))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename),map_location = "cpu"))
