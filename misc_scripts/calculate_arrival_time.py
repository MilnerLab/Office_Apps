import numpy as np

###############
# This program calculates the time of flight for a given m/q ratio
# (note that the constants are calculated using values which can change when changing the VMI voltages or the laser setup)
###############
def get_calib(t1,mq1,t2,mq2):
    a = (t1-t2)/(np.sqrt(mq1)-np.sqrt(mq2))
    t0 = 0.5*(t1+t2-a*(np.sqrt(mq1)+np.sqrt(mq2)))
    return (a,t0)

def get_time_delay(mq,a,t0):
    return a*np.sqrt(mq)+t0
    
###############
# time delay center of MCP and mass/charge values for VMI 6.04/3.99
###############

t_sulfur = 3747.0
mq_sulfur = 32.0

t_iodine = 3240
mq_iodine = 126.9

##For VMI 3/3.99
t_sulfur = 4400
t_cs2plus = 5565
mq_cs2plus = 76
a,t0 = get_calib(t_sulfur,mq_sulfur,t_cs2plus,mq_cs2plus)
mq = 42
t = get_time_delay(mq_iodine,380.6,2247)
print("a = ",round(a,2),", t0 = ",round(t0,2))
print("Time delay for m/q = ",mq_iodine," is t = ",round(t,2))
#print("Arrival time for m/q =", mq, ": ", round(calc_t, 2), "ns")

