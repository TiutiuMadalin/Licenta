import socket
import numpy as np


from DQN_Agent import DQN_Agent


backlog = 1
size = 1024
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('192.168.1.2', 50001))
s.listen(backlog)
state_size_ = 3
action_size_ = 2
dqn_agent = DQN_Agent(state_size_, action_size_)
dqn_agent.load('blackJack.h5')
dqn_agent.model.summary();
print("is waiting")
client, address = s.accept()
while 1:
    cur_state_ = client.recv(size)
    if cur_state_:
        cur_state_=cur_state_.decode("utf-8")
        list=cur_state_.split(", ")
        print(cur_state_)
        if(list[2]=="True"):
            cur_state_ = np.reshape((int(list[0]),int(list[1]),True), [1, dqn_agent.state_size])
        else:
            cur_state_ = np.reshape((int(list[0]), int(list[1]), False), [1, dqn_agent.state_size])
        action=dqn_agent.act(cur_state_)
        print(action)
        if action==1:
            to_send="hit"
        else:
            to_send="stand"
        arr=bytes(to_send,"utf-8")
        client.send(arr)


print("closing socket")
client.close()
s.close()