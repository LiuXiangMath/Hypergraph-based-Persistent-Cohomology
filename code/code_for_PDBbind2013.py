# -*- coding: utf-8 -*-


import numpy as np
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import scipy as sp

Protein_Atom = ['C','N','O','S']
Ligand_Atom = ['C','N','O','S','P','F','Cl','Br','I']
aa_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','HSE','HSD','SEC',
           'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','PYL']

pre = '' # this is the path where you place this file
Year = '2013'

f1 = open(pre + '../data/' + Year + '/name/train_data_' + Year + '.txt')
pre_train_data = f1.readlines()
train_data = eval(pre_train_data[0])
f1.close()

f1 = open(pre + '../data/' + Year + '/name/test_data_' + Year + '.txt')
pre_test_data = f1.readlines()
test_data = eval(pre_test_data[0])
f1.close()

f1 = open(pre + '../data/' + Year + '/name/all_data_' + Year + '.txt')
pre_all_data = f1.readlines()
all_data = eval(pre_all_data[0])
f1.close()



########################################################################################
# extract coordinate code starts
def get_index(a,b):
    t = len(b)
    if a=='Cl':
        #print('Cl')
        return 6
    if a=='Br':
        #print('Br')
        return 7
    
    for i in range(t):
        if a[0]==b[i]:
            return i
    return -1


def pocket_coordinate_data_to_file(start,end):
    #########################################################################
    '''
    this function extract the atom coordinates for each atom-pair of protein-ligand
    complex.
    output is a coordinate file and a description file, the description file records
    the number of atoms for protein and ligand. the coordinate file has four columns, 
    the former three columns are the coordinate, the last column are 1 and 2 for protein
    and ligand atoms respectively. 
    (1) start and end are index of data you will deal with
    (2) before this function, you need to prepare the PDBbind data
    '''
    #########################################################################
    t1 = len(all_data)
    for i in range(start,end):
        #print('process {0}-th '.format(i))
        
        protein = {}
        for ii in range(4):
            protein[Protein_Atom[ii]] = []
            
        name = all_data[i]
        t1 = pre + '../data/' + Year + '/refined/' + name + '/' + name + '_pocket.pdb'
        f1 = open(t1,'r')
        for line in f1.readlines():
            if (line[0:4]=='ATOM')&(line[17:20] in aa_list ):
                atom = line[13:15]
                atom = atom.strip()
                index = get_index(atom,Protein_Atom)
                if index==-1:
                    continue
                else:
                    protein[Protein_Atom[index]].append(line[30:54])
        f1.close()
        
        
        ligand = {}
        for ii in range(9):
            ligand[Ligand_Atom[ii]] = []
            
        t2 = pre + '../data/' +Year + '/refined/' + name + '/' + name + '_ligand.mol2'
        f2 = open(t2,'r')
        contents = f2.readlines()
        t3 = len(contents)
        start = 0
        end = 0
        for jj in range(t3):
            if contents[jj][0:13]=='@<TRIPOS>ATOM':
                start = jj + 1
                continue
            if contents[jj][0:13]=='@<TRIPOS>BOND':
                end = jj - 1
                break
        for kk in range(start,end+1):
            if contents[kk][8:17]=='thiophene':
                print('thiophene',kk)
            atom = contents[kk][8:10]
            atom = atom.strip()
            index = get_index(atom,Ligand_Atom)
            if index==-1:
                continue
            else:
                    
                ligand[Ligand_Atom[index]].append(contents[kk][17:46])
        f2.close()
        
        
        for i in range(4):
            for j in range(9):
                l_atom = ligand[ Ligand_Atom[j] ]
                p_atom = protein[ Protein_Atom[i] ]
                number_p = len(p_atom)
                number_l = len(l_atom)
                number_all = number_p + number_l
        
                all_atom = np.zeros((number_all,4))
                for jj in range(number_p):
                    all_atom[jj][0] = float(p_atom[jj][0:8])
                    all_atom[jj][1] = float(p_atom[jj][8:16])
                    all_atom[jj][2] = float(p_atom[jj][16:24])
                    all_atom[jj][3] = 1
                for jjj in range(number_p,number_all):
                    all_atom[jjj][0] = float(l_atom[jjj-number_p][0:9])
                    all_atom[jjj][1] = float(l_atom[jjj-number_p][9:19])
                    all_atom[jjj][2] = float(l_atom[jjj-number_p][19:29])
                    all_atom[jjj][3] = 2
        
                filename2 = pre + '../data/' + Year + '/pocket_coordinate/' + name + '_' + Protein_Atom[i] + '_' + Ligand_Atom[j] + '_coordinate.csv'
                np.savetxt(filename2,all_atom,delimiter=',')
                filename3 = pre + '../data/' + Year + '/pocket_coordinate/' + name +  '_' + Protein_Atom[i] + '_' + Ligand_Atom[j] + '_protein_ligand_number.csv'
                temp = np.array(([number_p,number_l]))
                np.savetxt(filename3,temp,delimiter=',')
        

#############################################################################################   
# extract coordinate code ends
                
                
                
                

#######################################################################################################
# create_the_associated_simplicial_complex_of_a_hypergraph algorithm starts 

def distance_of_two_point(p1,p2):
    s = pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2) + pow(p1[2]-p2[2],2)
    res = pow(s,0.5)
    return res
    

def get_edge_index(left,right,edges):
    t = len(edges)
    for i in range(t):
        if (left==edges[i][0])&(right==edges[i][1]):
            return i
    return -1


def create_simplices_with_filtration(atom,cutoff,name,P_atom,L_atom,kill_time):
    
    ##########################################################################################
    ''' 
    this function creates the filtered associated simplicial complex for the hypergraph.
    the dimension only up to 2. you can add higher dimensional information by adding some code. 
    (1) atom is the atom coordinates. the format is same with output of function 
        pocket_coordinate_to_file()
    (2) cutoff determines the binding core region we extract, that is, we extract the ligand
        atoms and the protein atoms within cutoff distance of the ligand. Here, cutoff also 
        determines the largest length of the edges we use to build the hypergraph, here also 
        the associated simplicial complex.(of course you can use many others methods to build 
        the complex, like you can add another parameter max_edge to control the largest length
        of an edge, this is just a way)
    (3) name is the data name.(for example, for PDBbind-2007, it has 1300 data, each data has 
        a name)
    (4) P_atom and L_atom are the atom-combination, like C-C, C-N, etc.
    (5) kill_time is an additional parameter, larger value will lead to longer persistence for
        all the barcode. here we use 0.
    (6) output is a sequence of ordered simplices, i.e. a filtered simplicial complex.
        the format for each simplex is as follows:
        [ index, filtration_value, dimension, vertices of the simplex ]
    '''
    ##########################################################################################
    
    vertices = []
    edge = []
    triangle = []
    edge_same_type = [] # edge_same_type stores the edges come from the same molecular. 
                        # i.e., the edges the hypergraph does not have.
    filtration_of_edge_same_type = []
        
    filename3 = pre + '../data/' + Year + '/pocket_coordinate/' + name + '_' + P_atom + '_' + L_atom +'_protein_ligand_number.csv'
    temp = np.loadtxt(filename3,delimiter=',') # temp gives the numbers of atoms for protein and ligand 
    number_p = int(temp[0])
    number_l = int(temp[1])
   
    t = atom.shape 
    atom_number = t[0] # t is equal to the sum of number_p and number_l
    if (number_p==0)|(number_l==0):# no complex
        return []
    
    for i in range(number_p):
        for j in range(number_p,atom_number):
            dis1 = distance_of_two_point(atom[i],atom[j])
            if dis1<=cutoff:    
                if ([i,j] in edge)==False:
                    edge.append([i,j])
                    if (i in vertices)==False:
                        vertices.append(i)
                    if (j in vertices)==False:
                        vertices.append(j)
                for k in range(atom_number):
                    if (k!=i)&(k!=j):
                        dis = -1
                        if atom[i][3]==atom[k][3]:
                            dis = distance_of_two_point(atom[j],atom[k])
                        else:
                            dis = distance_of_two_point(atom[i],atom[k])
                        
                        if dis<=cutoff:
                            One = 0
                            Two = 0
                            Three = 0
                            if k<i:
                                One = k
                                Two = i
                                Three = j
                            elif (k>i) & (k<j):
                                One = i
                                Two = k
                                Three = j
                            else:
                                One = i
                                Two = j
                                Three = k
                            if ([One,Two,Three] in triangle)==False:
                                triangle.append([One,Two,Three])
                                
                                if ([One,Two] in edge)==False:
                                    edge.append([One,Two])
                                    if atom[One][3]==atom[Two][3]:
                                        edge_same_type.append([One,Two])
                                        d1 = distance_of_two_point(atom[One],atom[Three])
                                        d2 = distance_of_two_point(atom[Two],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type.append(d)
                                else:
                                    edge_index = get_edge_index(One,Two,edge_same_type)
                                    if edge_index!=-1:
                                        temp = filtration_of_edge_same_type[edge_index]
                                        d1 = distance_of_two_point(atom[One],atom[Three])
                                        d2 = distance_of_two_point(atom[Two],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type[edge_index] = max(temp,d)
                            
                                if ([One,Three] in edge)==False:
                                    edge.append([One,Three])
                                    if atom[One][3]==atom[Three][3]:
                                        edge_same_type.append([One,Three])
                                        d1 = distance_of_two_point(atom[One],atom[Two])
                                        d2 = distance_of_two_point(atom[Two],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type.append(d)
                                else:
                                    edge_index = get_edge_index(One,Three,edge_same_type)
                                    if edge_index!=-1:
                                        temp = filtration_of_edge_same_type[edge_index]
                                        d1 = distance_of_two_point(atom[One],atom[Two])
                                        d2 = distance_of_two_point(atom[Two],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type[edge_index] = max(temp,d)
                                    
                                if ([Two,Three] in edge)==False:
                                    edge.append([Two,Three])
                                    if atom[Two][3]==atom[Three][3]:
                                        edge_same_type.append([Two,Three])
                                        d1 = distance_of_two_point(atom[One],atom[Two])
                                        d2 = distance_of_two_point(atom[One],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type.append(d)
                                else:
                                    edge_index = get_edge_index(Two,Three,edge_same_type)
                                    if edge_index!=-1:
                                        temp = filtration_of_edge_same_type[edge_index]
                                        d1 = distance_of_two_point(atom[One],atom[Two])
                                        d2 = distance_of_two_point(atom[One],atom[Three])
                                        d = max(d1,d2)
                                        filtration_of_edge_same_type[edge_index] = max(temp,d)
                                    
                                    
                                if (One in vertices)==False:
                                    vertices.append(One)
                                if (Two in vertices)==False:
                                    vertices.append(Two)
                                if (Three in vertices)==False:
                                    vertices.append(Three)
    
    
    for i in range(number_p,atom_number): # here, we add the ligand atoms we did not add in
        if (i in vertices)==False:
            vertices.append(i)
    
    vertices_number = len(vertices)
    edge_number = len(edge)
    triangle_number = len(triangle)
    simplices_with_filtration = []
    
    same_type_number = len(edge_same_type)
    for i in range(same_type_number):
        filtration_of_edge_same_type[i] = filtration_of_edge_same_type[i] + kill_time
   
    if vertices_number==0:
        return []
    for i in range(vertices_number):
        item = [ i , 0 , 0 , vertices[i] ]
        simplices_with_filtration.append(item)
    for i in range( vertices_number , vertices_number + edge_number ):
        one = edge[ i - vertices_number ][0]
        two = edge[ i - vertices_number ][1]
        p1 = atom[ one ]
        p2 = atom[ two ]
        dis = distance_of_two_point(p1,p2)
        edge_index = get_edge_index(one,two,edge_same_type)
        if edge_index!=-1:
            dis = filtration_of_edge_same_type[edge_index]
        dis = round(dis,15)
        if dis<=cutoff:
            item = [ i , dis , 1 , one , two ]
            simplices_with_filtration.append(item)
    for i in range( vertices_number + edge_number , vertices_number + edge_number + triangle_number ):
        one = triangle[ i - vertices_number - edge_number ][0]
        two = triangle[ i - vertices_number - edge_number ][1]
        three = triangle[ i - vertices_number - edge_number ][2]
        p1 = atom[ one ]
        p2 = atom[ two ]
        p3 = atom[ three ]
        dis = -1
        if ([one,two] in edge_same_type)==False:
            
            dis1 = distance_of_two_point(p1,p2)
            dis = max(dis,dis1)
        else:
            edge_index = get_edge_index(one,two,edge_same_type)
            temp = filtration_of_edge_same_type[edge_index]
            dis = max(dis,temp)
        if ([one,three] in edge_same_type)==False:
            
            dis2 = distance_of_two_point(p1,p3)
            dis = max(dis,dis2)
        else:
            edge_index = get_edge_index(one,three,edge_same_type)
            temp = filtration_of_edge_same_type[edge_index]
            dis = max(dis,temp)
        if ([two ,three] in edge_same_type)==False:
            
            dis3 = distance_of_two_point(p2,p3)
            dis = max(dis,dis3)
        else:
            edge_index = get_edge_index(two,three,edge_same_type)
            temp = filtration_of_edge_same_type[edge_index]
            dis = max(dis,temp)
        dis = round(dis,15)
        if dis<=cutoff:
            item = [ i , dis , 2 , one , two , three ]
            simplices_with_filtration.append(item)
    
    simplices = sorted(simplices_with_filtration,key=lambda x:(x[1]+x[2]/10000000000000000))
    # by applying the function sorted, the simplicies will be ordered by the filtration values.
    # also the face of a simplex will appear earlier than the simplex itself.
    
    for i in range(len(simplices)):
        simplices[i][0] = i # assign index for the ordered simplices
    return simplices


def simplices_to_file(start,end,cutoff,kill_time):
    ################################################################################################
    '''
    this function write the associated simplicial complex of the hypergraph to file
    (1) start and end are the indexes of data we deal with
    (2) cutoff, and kill_time are same with the function "create_simplices_with_filtration"
    (3) before this function, the function pocket_coordinate_data_to_file(start,end) need to 
    be performed to prepare the coordinate data for this function.
    '''
    ################################################################################################
    
    t = len(all_data)
    
    for i in range(start,end):
        name = all_data[i]
        print('process {0}-th data {1}'.format(i,name))
        for P in range(4):
            for L in range(9):
                filename = pre + '../data/' + Year + '/pocket_coordinate/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] +'_coordinate.csv'
                point_cloud = np.loadtxt(filename,delimiter=',')
                simplices_with_filtration = create_simplices_with_filtration(point_cloud,cutoff,name,Protein_Atom[P],Ligand_Atom[L],kill_time)
                filename2 = pre + '../data/' + Year + '/pocket_simplices_' + str(cutoff) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '.txt'
                f1 = open(filename2,'w')
                f1.writelines(str(simplices_with_filtration))
                f1.close()

                
                
######################################################################################################            
# create_the_associated_simplicial_complex_of_a_hypergraph algorithm ends
                




######################################################################################################
# the persistent cohomology algorithm starts from now(coefficient is Z/2)

def get_value_alpha_P_on_m_boundary(alpha_p,m_boundary,m_dimension):
    t1 = len(m_boundary)
    t2 = len(alpha_p)
    res = 0
    value = []
    
    for i in range(t1):
        value.append(0)
        for j in range(1,t2):
            if ( alpha_p[j][0:-1]==m_boundary[i]):
                value[i] = alpha_p[j][-1]
                break
    if m_dimension==0:
        res = 0
    elif m_dimension==1:
        res = value[1] - value[0]
    elif m_dimension==2:
        res = value[0] - value[1] + value[2]
    
    # can add more higher dimensional information you need
    if (res%2)==0:
        return 0
    return res


def delete_zero_of_base(base):
    t1 = len(base)
    new = [base[0]]
    for i in range(1,t1):
        if ((base[i][-1]%2)!=0):
            new.append(base[i])
    return new


def add_two_base_one_dimension(parameter1,base1,parameter2,base2):
    #############################################################
    '''
    this function compute the sum of parameter1*base1 and parameter2*base2
    base1 and base2 are both 1-cochain
    '''
    #############################################################
    t1 = len(base1)
    t2 = len(base2)
    b1 = np.ones((t1-1,3))
    b2 = np.ones((t2-1,3))
    
    for i in range(1,t1):
        b1[i-1][0] = base1[i][0]
        b1[i-1][1] = base1[i][1]
        b1[i-1][2] = base1[i][2]
    for i in range(1,t2):
        b2[i-1][0] = base2[i][0]
        b2[i-1][1] = base2[i][1]
        b2[i-1][2] = base2[i][2]
    count =t1-1 + t2-1
    
    for i in range(t1-1):
        for j in range(t2-1):
            if (b1[i][0]==b2[j][0])&(b1[i][1]==b2[j][1]):
                count = count -1
                break
    
    res = np.ones((count,3))
    for i in range(t1-1):
        b1[i][2] = b1[i][2]*parameter1
        res[i,:] = b1[i,:]
    C = t1 -1
    for i in range(t2-1):
        have = 0
        for j in range(t1-1):
            if (res[j][0]==b2[i][0])&(res[j][1]==b2[i][1]):
                res[j][2] = res[j][2]  + b2[i][2] * parameter2
                have = 1
                break
        if have ==0:
            b2[i][2] = b2[i][2] * parameter2
            res[C,:] = b2[i,:]
            C = C + 1
            
    rrr = [1]
    for i in range(count):
        rrr.append([res[i][0],res[i][1],res[i][2]])
    rrr = delete_zero_of_base(rrr) # only store nonzero information
    return rrr


def add_two_base_zero_dimension(parameter1,base1,parameter2,base2):
    #############################################################
    '''
    this function compute the sum of parameter1*base1 and parameter2*base2
    base1 and base2 are both 0-cochain
    '''
    #############################################################
    
    t1 = len(base1)
    t2 = len(base2)
    b1 = np.ones((t1-1,2))
    b2 = np.ones((t2-1,2))
    
    for i in range(1,t1):
        b1[i-1][0] = base1[i][0]
        b1[i-1][1] = base1[i][1]
       
    for i in range(1,t2):
        b2[i-1][0] = base2[i][0]
        b2[i-1][1] = base2[i][1]
       
    count =t1-1 + t2-1
    
    for i in range(t1-1):
        for j in range(t2-1):
            if (b1[i][0]==b2[j][0]):
                count = count -1
                break
            
    res = np.ones((count,2))
    for i in range(t1-1):
        b1[i][1] = b1[i][1] * parameter1
        res[i,:] = b1[i,:]
    C = t1 -1
    for i in range(t2-1):
        have = 0
        for j in range(t1-1):
            if (res[j][0]==b2[i][0]):
                res[j][1] = res[j][1]  + b2[i][1] * parameter2
                have = 1
                break
        if have ==0:
            b2[i][1] = b2[i][1] * parameter2
            res[C,:] = b2[i,:]
            C = C + 1
    
    
    rrr = [0]
    for i in range(count):
        rrr.append([res[i][0],res[i][1]])
    rrr = delete_zero_of_base(rrr) # only store nonzero information
    return rrr


def get_result(point_cloud,simplices_with_filtration):
    ######################################################################################
    '''
    this function generates the persistent cohomology barcodes and generators for the 
    associated simplicial complex of a hypergraph.
    (1) point_cloud is the coordinate data of a specific atom-combination of some data,
        the format is same with the output of pocket_coordinate_data_to_file()
    (2) simplicies_with_filtration is the output of function "create_simplices_with_filtration"
    (3) output is the zero_barcodes, zero_generators, one_barcodes and one_generators.
        you can get higher dimensional information by adding some code.
    '''
    ######################################################################################
    
    t1 = len(simplices_with_filtration)
    if t1==0:
        return []
    
    threshold = t1
    I = [0]
    P = []  # P is a list of pair[ [alpha_p,alpha_q],... ]  d(alpha_p) = alpha_q
    base = [ [0, [ int(simplices_with_filtration[0][3]) ,1]] ]
    # format of an element of base: [dimension , [simplices(increasing order),value]]
    
    for m in range(1,threshold):
        m_dimension = simplices_with_filtration[m][2]
        C = np.zeros((m,1))
        m_boundary = []
        if m_dimension==0:
            m_boundary.append([-1])
        elif m_dimension==1:
            m_boundary.append([simplices_with_filtration[m][3]])
            m_boundary.append([simplices_with_filtration[m][4]])
        elif m_dimension==2:
            zero_one = [simplices_with_filtration[m][3],simplices_with_filtration[m][4]]
            zero_two = [simplices_with_filtration[m][3],simplices_with_filtration[m][5]]
            one_two = [simplices_with_filtration[m][4],simplices_with_filtration[m][5]]
            m_boundary.append(zero_one)
            m_boundary.append(zero_two)
            m_boundary.append(one_two)
        
        # can add higher dimensional information if you need
        
        for p in P:
            alpha_p = base[p[0]]
            if (alpha_p[0] + 1)!= m_dimension:
                C[p[0]][0] = 0
            else:
                C[p[0]][0] = get_value_alpha_P_on_m_boundary(alpha_p,m_boundary,m_dimension)
            
            if C[p[0]][0]!=0:
                new_item = simplices_with_filtration[m][3:4+m_dimension] 
                new_item.append(C[p[0]][0])
                base[p[1]].append(new_item)
                
        I_max_none_zero_number = -100
        for  i in I:
            alpha_i = base[i]
            if (alpha_i[0] + 1)!= m_dimension:
                C[i][0] = 0
            else:
                C[i][0] = get_value_alpha_P_on_m_boundary(alpha_i,m_boundary,m_dimension)    
       
        for i in I:
            if (C[i][0]!=0)&(i>I_max_none_zero_number):
                I_max_none_zero_number = i
        
        if I_max_none_zero_number == -100:    
            I.append(m)
            new_item = [m_dimension]
            new_item.append(simplices_with_filtration[m][3:4+m_dimension])
            new_item[1].append(1)
            base.append(new_item)
                
        else:
            M = I_max_none_zero_number
            
            for t in range(len(I)):
                if I[t] == M:
                    del I[t]
                    break
            P.append([M,m])
           
            temp_base = [base[M][0]]
            for i in range(1,len(base[M])):
                temp_base.append(base[M][i])
            for i in I:
                
                if C[i][0]!=0:                
                    parameter = C[i][0]/C[M][0]
            
                    if (base[i][0]==0):
                        base[i] = add_two_base_zero_dimension(1,base[i],-parameter,temp_base)
                    elif base[i][0]==1:
                        base[i] = add_two_base_one_dimension(1,base[i],-parameter,temp_base)
                    # can add higher dimensional information if you need
                        
            new_item = [m_dimension]
            new_item.append(simplices_with_filtration[m][3:4+m_dimension])
            new_item[1].append(C[M][0])
            base.append(new_item)
            
    zero_cocycle = []
    one_cocycle =[]
    two_cocycle = []
    zero_bar = []
    one_bar = []
    two_bar = []
    
    
    for i in I:
        if base[i][0]==0:   
            zero_cocycle.append(base[i][1::])
            zero_bar.append([i,-1])
            
        elif base[i][0]==1: 
            one_cocycle.append(base[i][1::])
            one_bar.append([i,-1])
        # can add higher dimensional information if you need
    for p in P:
        
            if (base[p[0]][0]==0)&((simplices_with_filtration[p[1]][1]-simplices_with_filtration[p[0]][1])>0):  
                zero_cocycle.append(base[p[0]][1::])
                zero_bar.append([p[0],p[1]])
                
            elif (base[p[0]][0]==1)&((simplices_with_filtration[p[1]][1]-simplices_with_filtration[p[0]][1])>0):
                one_cocycle.append(base[p[0]][1::])
                one_bar.append([p[0],p[1]])
            # can add higher dimensional information if you need
   
    
    result = {'cocycles':[zero_cocycle,one_cocycle,two_cocycle],
              'diagrams':[zero_bar,one_bar,two_bar]}
    return result


def bar_and_cocycle_to_file(start,end,cutoff,filtration):
    ########################################################################################
    '''
    this function write the cohomology generators and barcodes to a file
    (1) start and end are the indexes of data we deal with
    (2) cutoff, and kill_time are same with the function "create_simplices_with_filtration"
    (3) parameter filtration determines the filtration range we use.
    (4) before this function, the function simplices_to_file(start,end,cutoff,kill_time) 
        should be performed to prepare the simplices data we use here
    '''
    ########################################################################################
    
    t = len(all_data)
    for i in range(start,end):
        name = all_data[i]
        print('process {0}-th  bar {1}'.format(i,name))
        for P in range(4):
            for L in range(9):
                filename1 = pre + '../data/' + Year + '/pocket_coordinate/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] +'_coordinate.csv'
                point_cloud = np.loadtxt(filename1,delimiter=',')
                filename2 = pre + '../data/' + Year + '/pocket_simplices_' + str(cutoff) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '.txt'
                f = open(filename2)
                pre_simplices = f.readlines()
                simplices = eval(pre_simplices[0])
                simplices_with_filtration = []
                for ii in range(len(simplices)):
                    if simplices[ii][1]<=filtration:
                        simplices_with_filtration.append(simplices[ii])
                    else:
                        break
                
                result = get_result(point_cloud,simplices_with_filtration)
                if result==[]:
                    filename1 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_zero_bar.csv'
                    zero_bar = np.zeros((1,2))
                    np.savetxt(filename1,zero_bar,delimiter=',')
                    
                    filename3 = pre + '../data/' + Year + '/pocket_cocycle_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_zero_cocycle.txt'
                    f3 = open(filename3,'w')
                    f3.writelines('')
                    f3.close()
                    
                    
                    filename2 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_one_bar.csv'
                    one_bar = np.zeros((1,2))
                    np.savetxt(filename2,one_bar,delimiter=',')
                    
                    filename4 = pre + '../data/' + Year + '/pocket_cocycle_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_one_cocycle.txt'
                    f4 = open(filename4,'w')
                    f4.writelines('')
                    f4.close()
                    
                    continue
                
                diagrams = result['diagrams']
                cocycles = result['cocycles']
                
                cocycle0 = cocycles[0]
                cocycle1 = cocycles[1]
                
                dgm0 = np.array(diagrams[0])
                dgm1 = np.array(diagrams[1])
        
                zero = dgm0.shape
                zero_number = zero[0]
                zero_bar = np.zeros((zero_number,2))
        
                one = dgm1.shape
                one_number = one[0]
                one_bar = np.zeros((one_number,2))
        
                for ii in range(zero_number):
                    left = dgm0[ii][0]
                    right = dgm0[ii][1]
                    zero_bar[ii][0] = simplices_with_filtration[left][1]
                    zero_bar[ii][1] = simplices_with_filtration[right][1]
                    if right==-1:
                        zero_bar[ii][1] = float('inf')
                for j in range(one_number):
                    left = dgm1[j][0]
                    right = dgm1[j][1]
                    one_bar[j][0] = simplices_with_filtration[left][1]
                    one_bar[j][1] = simplices_with_filtration[right][1]
                    if right==-1:
                        one_bar[j][1] = float('inf')
                
                
                #draw_barcodes(zero_bar,one_bar,max_distance)
                filename1 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_zero_bar.csv'
                np.savetxt(filename1,zero_bar,delimiter=',')
                filename3 = pre + '../data/' + Year + '/pocket_cocycle_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_zero_cocycle.txt'
                f3 = open(filename3,'w')
                f3.writelines(str(cocycle0))
                f3.close()
                
                filename2 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_one_bar.csv'
                np.savetxt(filename2,one_bar,delimiter=',')
                filename4 = pre + '../data/' + Year + '/pocket_cocycle_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_one_cocycle.txt'
                f4 = open(filename4,'w')
                f4.writelines(str(cocycle1))
                f4.close()
                


#######################################################################################################
# the persistent cohomology algorithm ends   
                
     



#####################################################################################################
# feature_generation algorithm starts from now

def get_number(bar,left,right):
    ##########################################################################
    '''
    this function compute the number of bars covering the interval [left,right]
    '''
    ##########################################################################
    t = bar.shape
    if (len(t)==1):
        return 0
    num = t[0]
    res = 0
    for i in range(num):
        if (bar[i][0]<=left)&(bar[i][1]>=right):
            res = res + 1
    return res


def get_feature_of_train(start,end,cutoff,filtration,unit):
    ##########################################################################
    '''
    this function generate the training feature vectors from HPC, the method 
    is bin counts.
    (1) cutoff and filtration are same with function "bar_and_cocycle_to_file"
    (2) unit is the size of each bin
    (3) before this function, function bar_and_cocycle_to_file() should be 
        performed to prepare the barcode
    '''
    ##########################################################################
    
    t = len(train_data)
    column0 = int( (filtration-2)/unit ) # start from 2
    column1 = int( (filtration-2)/unit )
    feature_matrix = np.zeros(( t , 36 * ( column0 + column1 ) ))
    
    for i in range(start,end):
        #print('process {0}-th of train feature'.format(i))
        name = train_data[i]
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_zero_bar.csv'
                zero_bar = np.loadtxt(filename0,delimiter=',')
                filename1 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_one_bar.csv'
                one_bar = np.loadtxt(filename1,delimiter=',')
                
                for n in range(column0):    
                    feature_matrix[i][count] = get_number( zero_bar , 2 + unit * n , 2 + unit * (n+1) )
                    count = count + 1
                for n in range(column1):    
                    feature_matrix[i][count] = get_number( one_bar , 2 + unit * n , 2 + unit * (n+1) )
                    count = count + 1
                
    #draw_barcodes(zero_bar,one_bar)
    np.savetxt(pre + '../data/' + Year + '/pocket_feature/feature_matrix_of_train_36_' + str(cutoff) + '_' + str(filtration) + '_' + str(unit) + '.csv',feature_matrix,delimiter=',')
    
def get_feature_of_test(start,end,cutoff,filtration,unit):
    ##########################################################################
    '''
    this function generate the testing feature vectors from HPC, the method 
    is bin counts.
    (1) cutoff and filtration are same with function "bar_and_cocycle_to_file"
    (2) unit is the size of each bin
    (3) before this function, function bar_and_cocycle_to_file() should be
        performed to prepare the barcode
    '''
    ##########################################################################
    
    t = len(test_data)
    column0 = int( (filtration-2)/unit ) # start from 2
    column1 = int( (filtration-2)/unit )
    feature_matrix = np.zeros(( t , 36 * ( column0 + column1 ) ))
    
    for i in range(start,end):
        #print('process {0}-th of test feature'.format(i))
        name = test_data[i]
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_zero_bar.csv'
                zero_bar = np.loadtxt(filename0,delimiter=',')
                filename1 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_one_bar.csv'
                one_bar = np.loadtxt(filename1,delimiter=',')
                
                for n in range(column0):    
                    feature_matrix[i][count] = get_number( zero_bar , 2 + unit * n , 2 + unit * (n+1) )
                    count = count + 1
                for n in range(column1):    
                    feature_matrix[i][count] = get_number( one_bar , 2 + unit * n , 2 + unit * (n+1) )
                    count = count + 1
                
    #draw_barcodes(zero_bar,one_bar)
    np.savetxt(pre + '../data/' + Year + '/pocket_feature/feature_matrix_of_test_36_' + str(cutoff) + '_' + str(filtration) + '_' + str(unit) + '.csv',feature_matrix,delimiter=',')
    






def get_name_index(name,contents):
    t = len(contents)
    for i in range(t):
        if contents[i][0:4]==name:
            return i

           
def get_target_matrix_of_train():
    t = len(train_data)
    target_matrix = []
    t1 = pre + '../data/' + Year + '/' + Year + '_INDEX_refined.data'
    f1 = open(t1,'r')
    contents = f1.readlines()
    f1.close()
    for i in range(t):  # tttttttttttttttttttttttttttttttttt
        name = train_data[i]
        index = get_name_index(name,contents)
        target_matrix.append(float(contents[index][18:23]))
    res = np.array(target_matrix)
    np.savetxt(pre + '../data/' + Year + '/pocket_feature/' + 'target_matrix_of_train.csv',res,delimiter=',')


def get_target_matrix_of_test():
    t = len(test_data)
    target_matrix = []
    t1 = pre + '../data/' + Year + '/' + Year + '_INDEX_refined.data'
    f1 = open(t1,'r')
    contents = f1.readlines()
    f1.close()
    for i in range(t):  # tttttttttttttttttttttttttttttttttt
        name = test_data[i]
        index = get_name_index(name,contents)
        target_matrix.append(float(contents[index][18:23]))
    res = np.array(target_matrix)
    np.savetxt(pre + '../data/' + Year + '/pocket_feature/' + 'target_matrix_of_test.csv',res,delimiter=',')
    




     
def create_coordinate_with_associated_distance(start,end):
    ######################################################################################
    '''
    this function compute all the adjacent distances from a atom to its all adjacent atoms.
    then, these distance will be used to form the centrality weight for each atom.
    '''
    ######################################################################################
    pre1 = pre + '../data/' + Year + '/pocket_coordinate/'
    pre2 = pre + '../data/' + Year + '/pocket_coordinate_with_associated_distance/'
    length = len(all_data)
    for i in range(start,end):
        print('process: ',i)
        name = all_data[i]
        for P in range(4):
            for L in range(9):
                filename1 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'coordinate.csv'
                data1 = np.loadtxt(filename1,delimiter=',')
                #s1 = data1.shape
                #row = s1[0]
                filename2 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'protein_ligand_number.csv'
                temp = np.loadtxt(filename2,delimiter=',')
                number_p = int(temp[0])
                number_l = int(temp[1])
                row = number_p + number_l
                column = max(number_p,number_l) + 4
                data2 = np.zeros((row,column))
                if (number_p==0) | (number_l==0):
                    filename3 = pre2 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'coordinate.csv'
                    np.savetxt(filename3,data2,delimiter=',')
                    continue
                for ii in range(0,number_p):
                    data2[ii][0:4] = data1[ii,::]
                    for j in range(4,4+number_l):
                        dis = distance_of_two_point(data1[ii],data1[number_p+j-4])
                        data2[ii][j] = dis
                for ii in range(number_p,number_p+number_l):
                    data2[ii][0:4] = data1[ii,::]
                    for j in range(4,4+number_p):
                        dis = distance_of_two_point(data1[ii],data1[j-4])
                        data2[ii][j] = dis
                        
                filename3 = pre2 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'coordinate.csv'
                np.savetxt(filename3,data2,delimiter=',')
                
                

          


def get_cocycle_feature_value0_centrality(cutoff,filtration,name,P,L,bar,left,right,eta):
    ######################################################################################
    '''
    this function get the sum of values of the enriched 0-barcodes in interval [left,right]
    (1) cutoff and filtration are same with function "bar_and_cocycle_to_file"
    (2) name is the data name
    (3) P and L is the atom names for atom-pair
    (4) bar is the 0-cohomology barcodes
    (5) eta is the parameter control the region we capture
    '''
    ######################################################################################
    t = bar.shape
    if (len(t)==1):
        return 0
    
    filename1 = pre + '../data/' + Year + '/pocket_cocycle_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + P + '_' + L + '_' + str(cutoff) + '_' + str(filtration) + '_zero_cocycle.txt'
    f1 = open(filename1)
    pre_zero_cocycle = f1.readlines()
    zero_cocycle = eval(pre_zero_cocycle[0])
    f1.close()
    
    filename2 = pre + '../data/' + Year + '/pocket_coordinate_with_associated_distance/' + name + '_' + P + '_' + L +'_coordinate.csv'
    point_cloud = np.loadtxt(filename2,delimiter=',')
    p_shape = point_cloud.shape
    num = t[0]
    res = 0
    for i in range(num):
        if (bar[i][0]<=left)&(bar[i][1]>=right):
            cocycle = zero_cocycle[i]
            t2 = len(cocycle)
            res2 = 0
            for j in range(t2):
                 one = int(cocycle[j][0])
                 value = abs(cocycle[j][1]) # coefficient is Z/2, -1==1
                 temp_weight = 0
                 for inner in range(4,p_shape[1]):
                     if point_cloud[one][inner]==0:
                         break
                     frac = pow(point_cloud[one][inner]/eta,2)
                     v = math.exp(-frac)
                     temp_weight = temp_weight + v
                          
                 res2 = res2 + value * temp_weight
            res = res + res2/t2
    
    return res  





def get_cocycle_feature_value1_centrality(cutoff,filtration,name,P,L,bar,left,right,eta):
    ######################################################################################
    '''
    this function get the sum of values of the enriched 1-barcodes in interval [left,right]
    (1) cutoff and filtration are same with function "bar_and_cocycle_to_file"
    (2) name is the data name
    (3) P and L is the atom names for atom-pair
    (4) bar is the 1-cohomology barcodes
    (5) eta is the parameter control the region we capture
    '''
    ######################################################################################
    
    t = bar.shape
    if (len(t)==1):
        return 0
    
    filename1 = pre + '../data/' + Year + '/pocket_cocycle_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + P + '_' + L + '_' + str(cutoff) + '_' + str(filtration) + '_one_cocycle.txt'
    f1 = open(filename1)
    pre_one_cocycle = f1.readlines()
    one_cocycle = eval(pre_one_cocycle[0])
    f1.close()
    
    filename2 = pre + '../data/' + Year + '/pocket_coordinate/' + name + '_' + P + '_' + L +'_coordinate.csv'
    point_cloud = np.loadtxt(filename2,delimiter=',')
    
    num = t[0]
    res = 0
    count = 0
    
    for i in range(num):
        if (bar[i][0]<=left)&(bar[i][1]>=right):
            cocycle = one_cocycle[i]
            t2 = len(cocycle)
            res2 = 0
            
            for j in range(t2):
                 one = int(cocycle[j][0])
                 two = int(cocycle[j][1])
                 value = abs(cocycle[j][2])
                 dis = distance_of_two_point(point_cloud[one],point_cloud[two]) 
                 frac = pow(dis/eta,2)
                 v = math.exp(-frac)
                 res2 = res2 + value * v
            res = res + res2/t2
            
    return res
    

def get_cocycle_feature_of_train(start,end,cutoff,filtration,unit,eta):
    #######################################################################################
    '''
    this function generate the training feature vectors from HWPC, the method is bin counts.
    (1) start and end are the indexes of the data we deal with
    (2) cutoff and filtration are same with function "bar_and_cocycle_to_file"
    (3) unit is the size of each bin
    (4) eta is the parameter for weight
    (5) before this funcition, function create_coordinate_with_associated_distance() should 
        be performed.
    '''
    #######################################################################################
    
    t = len(train_data)

    column0 = int((filtration - 2)/unit  )
    column1 = int((filtration - 2)/unit)
    
    column_cocycle0 = int( (filtration - 2)/unit )
    column_cocycle1 = int( (filtration - 2)/unit )
    
    feature_matrix = np.zeros(( end - start , 36 * ( column0 + column1 + column_cocycle0 + column_cocycle1 ) ))
    
    for i in range(start,end):
        name = train_data[i]
        #print('process {0}-th of train feature,{1}'.format(i,name))
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_zero_bar.csv'
                zero_bar = np.loadtxt(filename0,delimiter=',')
                filename1 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_one_bar.csv'
                one_bar = np.loadtxt(filename1,delimiter=',')
                for n in range(column0):
                    feature_matrix[i-start][count] = get_number( zero_bar , 2 + unit * n , 2 + unit * (n + 1) )
                    count = count + 1
                
                for n in range(column_cocycle0):
                    feature_matrix[i-start][count] = get_cocycle_feature_value0_centrality(cutoff,filtration,name,Protein_Atom[P],Ligand_Atom[L],zero_bar,2 + unit * n, 2 + unit * (n+1),eta)
                    count = count + 1
                
                for n in range(column1):
                    feature_matrix[i-start][count] = get_number( one_bar , 2 + unit * n  , 2 + unit * (n+1) )
                    count = count + 1
                    
                for n in range(column_cocycle1):
                    feature_matrix[i-start][count] = get_cocycle_feature_value1_centrality(cutoff,filtration,name,Protein_Atom[P],Ligand_Atom[L],one_bar,2 + unit * n, 2 + unit * (n+1),eta)
                    count = count + 1
                        
                    
    np.savetxt(pre + '../data/' + Year + '/pocket_feature/eta_' + str(eta) + '_cocycle_feature_matrix_of_train_36_' + str(cutoff) + '_' + str(filtration) + '_' + str(unit) + '.csv',feature_matrix,delimiter=',')
    


def get_cocycle_feature_of_test(start,end,cutoff,filtration,unit,eta):
    ######################################################################################
    '''
    this function generate the testing feature vectors from HWPC, the method is bin counts.
    (1) start and end are the indexes of the data we deal with
    (2) cutoff and filtration are same with function "bar_and_cocycle_to_file"
    (3) unit is the size of each bin
    (4) eta is the parameter for weight
    (5) before this funcition, function create_coordinate_with_associated_distance() should 
        be performed.
    '''
    ######################################################################################
    
    t = len(test_data)

    column0 = int((filtration - 2)/unit  )
    column1 = int((filtration - 2)/unit)
    
    column_cocycle0 = int( (filtration - 2)/unit )
    column_cocycle1 = int( (filtration - 2)/unit )
    
    feature_matrix = np.zeros(( end - start , 36 * ( column0 + column1 + column_cocycle0 + column_cocycle1 ) ))
    
    for i in range(start,end):
        name = test_data[i]
        #print('process {0}-th of test feature,{1}'.format(i,name))
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_zero_bar.csv'
                zero_bar = np.loadtxt(filename0,delimiter=',')
                filename1 = pre + '../data/' + Year + '/pocket_bar_' + str(cutoff) + '_' + str(filtration) + '/' + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + str(cutoff) + '_' + str(filtration) + '_one_bar.csv'
                one_bar = np.loadtxt(filename1,delimiter=',')
                for n in range(column0):
                    feature_matrix[i-start][count] = get_number( zero_bar , 2 + unit * n , 2 + unit * (n + 1) )
                    count = count + 1
                
                for n in range(column_cocycle0):
                    feature_matrix[i-start][count] = get_cocycle_feature_value0_centrality(cutoff,filtration,name,Protein_Atom[P],Ligand_Atom[L],zero_bar,2 + unit * n, 2 + unit * (n+1),eta)
                    count = count + 1
                
                for n in range(column1):
                    feature_matrix[i-start][count] = get_number( one_bar , 2 + unit * n  , 2 + unit * (n+1) )
                    count = count + 1
                    
                for n in range(column_cocycle1):
                    feature_matrix[i-start][count] = get_cocycle_feature_value1_centrality(cutoff,filtration,name,Protein_Atom[P],Ligand_Atom[L],one_bar,2 + unit * n, 2 + unit * (n+1),eta)
                    count = count + 1
                        
                    
    np.savetxt(pre + '../data/' + Year + '/pocket_feature/eta_' + str(eta) + '_cocycle_feature_matrix_of_test_36_' + str(cutoff) + '_' + str(filtration) + '_' + str(unit) + '.csv',feature_matrix,delimiter=',')
    



    


def get_combined_feature(typ,cutoff,filtration,unit):
    #####################################################################
    '''
    this function get the combined feature vectors from HWPC with 
    a lower eta 2.5 and another HWPC with a higher eta 10
    '''
    #####################################################################
    
    filename1 = pre + '../data/' + Year + '/pocket_feature/' + 'eta_2.5_cocycle_feature_matrix_of_' + typ + '_36_' + str(cutoff) + '_' + str(filtration) + '_' + str(unit) + '.csv'
    filename2 = pre + '../data/' + Year + '/pocket_feature/' + 'eta_10_cocycle_feature_matrix_of_' + typ + '_36_' + str(cutoff) + '_' + str(filtration) + '_' + str(unit) + '.csv'
    m1 = np.loadtxt(filename1,delimiter=',')
    m2 = np.loadtxt(filename2,delimiter=',')
    t1 = m1.shape
    t2 = m2.shape
    number = int((filtration-2)/0.1)
    m = np.zeros((t1[0],36*number*2*3))
    for i in range(t1[0]):
        for j in range(36):
            m[i][j*number*6:j*number*6+number*2] = m1[i][j*number*4:j*number*4+number*2]
            m[i][j*number*6+number*2:j*number*6+number*3] = m2[i][j*number*4+number:j*number*4+number*2]
            m[i][j*number*6+number*3:j*number*6+number*5] = m1[i][j*number*4+number*2:j*number*4+number*4]
            m[i][j*number*6+number*5:j*number*6+number*6] = m2[i][j*number*4+number*3:j*number*4+number*4]
            
    filename3 = pre + '../data/' + Year + '/pocket_feature/' + 'mix_eta_2.5_10_cocycle_feature_matrix_of_' + typ + '_36_' + str(cutoff) + '_' + str(filtration) + '_' + str(unit) + '.csv'
    np.savetxt(filename3,m,delimiter=',')
    
 
    
     
############################################################################################################
# feature_generation algorithm ends.

    
    
    
    
############################################################################################################
# machine_learning algorithm starts.
    
def gradient_boosting(X_train,Y_train,X_test,Y_test):
    params={'n_estimators': 40000, 'max_depth': 9, 'min_samples_split': 2,
                'learning_rate': 0.001, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
    regr = GradientBoostingRegressor(**params)
    regr.fit(X_train,Y_train)
    pearson_coorelation = sp.stats.pearsonr(Y_test,regr.predict(X_test))
    mse1 = mean_squared_error(Y_test, regr.predict(X_test))
    mse2 = pow(mse1,0.5)
    #mse3 = mse2/0.7335
    mse3 = mse2
    return [pearson_coorelation[0],mse3]

def get_pearson_correlation(typ,pref):
    feature_matrix_of_train = np.loadtxt( pre + '../data/' + Year + '/pocket_feature/' + pref +'feature_matrix_of_train_36_10.5_7.5_0.1.csv',delimiter=',' )
    target_matrix_of_train = np.loadtxt( pre + '../data/' + Year + '/pocket_feature/' + 'target_matrix_of_train.csv',delimiter=',' )
    feature_matrix_of_test = np.loadtxt( pre + '../data/' + Year + '/pocket_feature/' + pref + 'feature_matrix_of_test_36_10.5_7.5_0.1.csv',delimiter=',' )
    target_matrix_of_test = np.loadtxt( pre + '../data/' + Year + '/pocket_feature/' +  'target_matrix_of_test.csv',delimiter=',' )
    number = 10
    P = np.zeros((number,1))
    M = np.zeros((number,1))
    #print(feature_matrix_of_test.shape)
    for i in range(number):
        [P[i][0],M[i][0]] = gradient_boosting(feature_matrix_of_train,target_matrix_of_train,feature_matrix_of_test,target_matrix_of_test)
    median_p = np.median(P)
    median_m = np.median(M)
    print('for data ' + Year + ', 10 results for ' + typ + '-model are:')
    print(P)
    print('median pearson correlation values are')
    print(median_p)
    print('median mean squared error values are')
    print(median_m)
    
    
############################################################################################################
# machine_learning algorithm ends.

    
    
    
    
def run_for_PDBbind_2013():
    ##############################################################
    '''
    by running this function, you can get the results for data2013
    (1) before run this function, you should change the parameter
        Year to '2013'
    '''
    ##############################################################
    # extract coordinate
    pocket_coordinate_data_to_file(0,2959) 
    
    # create hypergraph
    simplices_to_file(0,2959,10.5,0)      
    
    # compute persistent cohomology
    bar_and_cocycle_to_file(0,2959,10.5,7.5) 
    
    # feature generation
    get_feature_of_train(0,2764,10.5,7.5,0.1) 
    get_feature_of_test(0,195,10.5,7.5,0.1)
    get_target_matrix_of_train()
    get_target_matrix_of_test()
    create_coordinate_with_associated_distance(0,2959)    
    get_cocycle_feature_of_train(0,2764,10.5,7.5,0.1,2.5)
    get_cocycle_feature_of_test(0,195,10.5,7.5,0.1,2.5)    
    get_cocycle_feature_of_train(0,2764,10.5,7.5,0.1,10)
    get_cocycle_feature_of_test(0,195,10.5,7.5,0.1,10)
    get_combined_feature('train',10.5,7.5,0.1)
    get_combined_feature('test',10.5,7.5,0.1) 
    
    # machine learning
    get_pearson_correlation('HPC','')
    get_pearson_correlation('HWPC2.5','eta_2.5_cocycle_')
    get_pearson_correlation('HWPC10','eta_10_cocycle_')
    get_pearson_correlation('combined', 'mix_eta_2.5_10_cocycle_')
    

    
run_for_PDBbind_2013()









