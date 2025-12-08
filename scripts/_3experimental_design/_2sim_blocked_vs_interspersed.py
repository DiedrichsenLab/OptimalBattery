from numpy.linalg import inv
import matplotlib.pyplot as plt
import os
import OptimalBattery.design as ds


## Make and show design matrices for the two designs 
design = [[[1, 2], [3,4]],  # Blocked design
        [[1, 2, 3, 4]] * 2]  # Interspersed design
title = ['Blocked','Interspersed']

plt.figure(figsize=(10, 5))
for i in range(2):
    X, cond_v, part_v, reg_ind, part_ind, lc, lr = ds.make_design_matrix(design[i],
                                                                      p_rest=0.2,
                                                                      T=200,
                                                                      num_rest=2,
                                                                      instruction_TR=0,
                                                                      randomize_order=True)
    plt.subplot(1, 2, i + 1)
    plt.title('Design %d' % (i + 1))
    plt.imshow(X,aspect='auto')
    plt.imshow(X,aspect='auto',interpolation='none')
    ds.draw_reference_lines('hor', part_v, color='k', lw=1)
    ds.draw_reference_lines('ver', part_ind, color='k', lw=1)
    pass


# For checking - this is covariance matrix (blocked)
design_b=[[1,2],[3,4]] 
X1,_,_,reg_ind,part_ind,_,_ = ds.make_design_matrix(design_b)
print(reg_ind)
conv_beta = inv(X1.T@X1)
plt.imshow(conv_beta)
ds.draw_reference_lines('hor', part_ind, color='k', lw=2)
ds.draw_reference_lines('ver', part_ind, color='k', lw=2)

# This is the covariance matrix for the regressors (interspersed)
design_i=[[1,2,3,4]]*2 
X2,_,_,reg_ind,part_ind,_,_ = ds.make_design_matrix(design_i)
conv_beta = inv(X2.T@X2)
plt.imshow(conv_beta)
ds.draw_reference_lines('hor', part_ind, color='k', lw=2)
ds.draw_reference_lines('ver', part_ind, color='k', lw=2)



# Now we are calculating the std of the
# contrast against rest
# contrast of different conditions (within runs)
# contrast of different conditions (between runs)

T=200
numpart = len(design_b)
DF = ds.compare_designs_1([design_b,design_i],T=T, instruction_time=0)
save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'design_simulation.tsv')
DF.to_csv(save_path, sep='\t', index=False)


#---- now do the same removing 5 instruction time points from interspersed design

# For checking - this is covariance matrix between regressors (blocked)
X1,_,_,reg_ind,part_ind,_,_ = ds.make_design_matrix(design_b,T=200,p_rest=0.2,num_rest=2,instruction_TR=0)
conv_beta = inv(X1.T@X1)
plt.imshow(conv_beta)
ds.draw_reference_lines('hor', part_ind, color='k', lw=2)
ds.draw_reference_lines('ver', part_ind, color='k', lw=2)

# THis is the covariance matrix for the regressors (interspersed)
X2,_,_,reg_ind,part_ind,_,_ = ds.make_design_matrix(design_i,T=200,p_rest=0.2,num_rest=1,instruction_TR=5)
conv_beta = inv(X2.T@X2)
plt.imshow(conv_beta)
ds.draw_reference_lines('hor', part_ind, color='k', lw=2)
ds.draw_reference_lines('ver', part_ind, color='k', lw=2)

T=200
numpart = len(design_b)
DF = ds.compare_designs_1([design_b,design_i],T=T, instruction_time=5)
save_path = os.path.join(save_dir, 'design_simulation_instruction.tsv')
DF.to_csv(save_path, sep='\t', index=False)
