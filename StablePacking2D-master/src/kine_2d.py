from matplotlib import collections as mc
import matplotlib.pyplot as plt
# import parameters
import numpy as np
import sys
import mosek
import math
import random
print_detail = False
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(1000)]


def cal_Aglobal(elems, contps):

    def cal_A_local(element, point, reverse=False):
        t = np.array(point.orientation[0])
        n = np.array(point.orientation[1])
        if reverse:
            t = t*(-1)
            n = n*(-1)

        R = - np.array(point.coor) + np.array(element.center)
        Alocal = np.matrix([
            [-1*t[0], -1*n[0]],
            [-1*t[1], -1*n[1]],
            [-1*float(np.cross(R, t)), -1*float(np.cross(R, n))]
        ])
        return Alocal

    Aglobal = np.zeros((3*len(elems), 2*len(contps)))
    row = 0
    for element in elems.values():
        col = 0
        for p_c in contps.values():
            if p_c.cand == element.id:
                Alocal = cal_A_local(element, p_c)
                Aglobal[row:row+3, col:col+2] = Alocal
            elif p_c.anta == element.id:
                Alocal = cal_A_local(element, p_c, reverse=True)
                Aglobal[row:row+3, col:col+2] = Alocal
            col += 2
        row += 3
    return Aglobal


def normalize(v):
    norm = np.linalg.norm(v, ord=2)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm



class ContPoint():
    def __init__(self, id, coor, cand, anta, t, n, cont_type, section_h=None, lever=None, faceID=-1):
        self.id = id
        self.coor = coor
        self.cand = cand
        self.anta = anta
        self.orientation = [t, n]
        self.cont_type = cont_type

        # for limbasd compression strength
        if cont_type == 'friction_fc':
            # give erros message if section_h or lever is not defined
            if section_h is None:
                raise Exception(
                    'section_h is not defined for friction_fc contact point')
            if lever is None:
                raise Exception(
                    'lever is not defined for friction_fc contact point')
            if lever < 0:
                raise Exception(
                    f'lever is negative for friction_fc contact point ({self})')

        self.section_h = section_h
        self.lever = lever
        self.faceID = faceID
        self.conterPoint = None

        # for non-associative solution
        self.normal_force = None
        self.c0 = None

        # record result
        self.displacement = [0, 0]

    def set_elements(self, elem1, elem2):
        self.cand = elem1
        self.anta = elem2

    def __eq__(self, other):
        if self.coor == other.coor:
            if self.orientation == other.orientation:
                return True
        return False

    def __str__(self):
        return f"Contact point information: \nID: {self.id}\nCandidate id: {self.cand}\nAntagonist id: {self.anta}"

    # def is_contat_pair(self, other):
    #     # two points are paired if they have the same coordianates and reversed normals
    #     if np.allclose(np.array(self.coor), np.array(other.coor), rtol=1e-5):
    #         if np.allclose(np.array(self.orientation[1]), np.array(other.orientation[1])*-1, rtol=1e-5):
    #             return True
    #     return False
    def is_contat_pair(self, other):
        # two points are paired if they have the same coordianates and reversed normals
        if np.allclose(np.array(self.coor), np.array(other.coor), rtol=1e-5):
            if self.cand != other.cand and np.array(self.orientation[1])@np.array(other.orientation[1]) < 0:
                # if np.allclose(np.array(self.orientation[1]), np.array(other.orientation[1])*-1, rtol=1e-5):
                return True
        return False

    def assert_legal(self):
        if self.id == -1:
            raise Exception(f'undefined contact point {self.id}')

        if not self.coor:
            raise Exception(f'contact point {self.id} undefined coordinates')

        if self.cand == -1 or self.anta == -1:
            raise Exception(
                f'contact point {self.id} undefined contact elements')
        if self.cont_type == "friction_fc" and self.faceID == -1:
            raise Exception(
                f'contact point {self.id} undefined face id')


class ContType():
    def __init__(self, name, paras):
        self.name = name
        if name == 'friction':
            self.mu = paras.mu
        elif name == 'friction_fc':
            self.mu = paras.mu
            self.fc = paras.fc
        elif name == 'friction_fc_cohesion':
            self.mu = paras.mu
            self.fc = paras.fc
            self.cohesion = paras.cohesion
        elif name == 'friction_cohesion':
            self.mu = paras.mu
            self.cohesion = paras.cohesion
        else:
            raise Exception('unknown contact type')


class Element():
    def __init__(self, id, center, mass, vertices, type='None'):
        self.id = id
        self.center = center
        self.mass = mass
        self.vertices = vertices
        self.dl = []
        self.ll = []
        self.type = type

        self.displacement = [0, 0, 0]

    def assert_legal(self, solver='rigid-associative'):
        if self.id == -1:
            raise Exception(f'undefined element {self.id}')

        if self.mass <= 0:
            raise Exception(f'element{self.id} has negative/zero mass')
        # check load
        if not self.dl:
            raise Exception(f'element{self.id} dead load not defined')
        if not self.ll:
            raise Exception(f'element{self.id} live load not defined')


class ContFace():
    def __init__(self, id, height, cont_type):
        self.id = id
        self.contps = []
        self.height = height
        if cont_type.name == "friction_fc" or cont_type.name == "friction_fc_cohesion":
            self.fc = cont_type.fc

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False


def solve_force_rigid(elems, contps, Aglobal):
    result = dict()
    inf = 0.0

    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    limit_force = 0
    # Make mosek environment
    with mosek.Env() as env:
        # Create a task object
        with env.Task(0, 0) as task:
            # Attach a log stream printer to the task
            if print_detail:
                task.set_Stream(mosek.streamtype.log, streamprinter)

            # Bound keys and values for constraints -- force equilibrium
            bkc = []
            blc = []
            buc = []
            elem_index = 0
            for key, value in elems.items():
                if value.type == "ground":
                    bkc.extend([mosek.boundkey.fr,
                                mosek.boundkey.fr,
                                mosek.boundkey.fr])
                    blc.extend([-inf, -inf, -inf])
                    buc.extend([inf, inf, inf])
                else:
                    bkc.extend([mosek.boundkey.fx,
                                mosek.boundkey.fx,
                                mosek.boundkey.fx])
                    blc.extend([value.dl[0], value.dl[1], value.dl[2]])
                    buc.extend([value.dl[0], value.dl[1], value.dl[2]])

            # Bound keys and values for constraints -- contact failure condition
            for key, value in contps.items():
                if value.cont_type.name == 'friction':
                    for i in range(3):
                        bkc.append(mosek.boundkey.up)
                        blc.append(-inf)
                        buc.append(0.0)
                else:
                    raise NameError("unknown contact type!")

            # Bound keys for variables
            bkx = []
            blx = []
            bux = []
            g_index = 0
            for key, value in contps.items():
                for i in range(2):
                    bkx.append(mosek.boundkey.fr)
                    blx.append(-inf)
                    bux.append(+inf)

                    g_index += 1
            bkx.append(mosek.boundkey.lo)
            blx.append(0)
            bux.append(+inf)

            # Objective coefficients
            c = []
            g_index = 0
            for key, value in contps.items():
                for i in range(2):  # 2variables(t,n)*1nodes*2contact faces
                    c.append(-0)
                    # print(-g[g_index])
                    g_index += 1
            c.append(1.0)

            # Below is the sparse representation of the A
            # matrix stored by column.
            asub = []
            aval = []
            for i, value in enumerate(contps.values()):
                for j in range(2):  # 2variables(t,n)*1nodes*
                    col = i*2+j
                    col_index = []
                    col_value = []
                    for row in range(len(elems)*3):
                        if Aglobal[row][col] != 0:
                            col_index.append(row)
                            col_value.append(Aglobal[row][col])
                    _start_row = len(elems)*3 + math.floor(col/2)*3
                    col_index.extend(
                        list(range(_start_row, _start_row+3)))
                    if col % 2 == 0:
                        col_value.extend([1, -1, 0])
                    else:
                        col_value.extend(
                            [-value.cont_type.mu, -value.cont_type.mu, -1])
                    asub.append(col_index)
                    aval.append(col_value)

            col_index = []
            col_value = []
            i = 0
            for key, value in elems.items():
                col_index.extend([3*i, 3*i+1, 3*i+2])
                col_value.extend(
                    [-value.ll[0], -value.ll[1], -value.ll[2]])
                i += 1
            asub.append(col_index)
            aval.append(col_value)

            numvar = len(bkx)
            numcon = len(bkc)

            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(numcon)

            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numvar)

            for j in range(numvar):
                # Set the linear term c_j in the objective.

                task.putcj(j, c[j])

                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(j, bkx[j], blx[j], bux[j])

                # Input column j of A
                task.putacol(j,                  # Variable (column) index.
                             # Row index of non-zeros in column j.
                             asub[j],
                             aval[j])            # Non-zero Values of column j.

            # Set the bounds on constraints.
             # blc[i] <= constraint_i <= buc[i]

            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)

            # Solve the problem
            task.optimize()
            if print_detail:
                task.writedata("./data.opf")
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.msg)

            # Get status information about the solution
            solsta = task.getsolsta(mosek.soltype.bas)

            xx = [0.] * numvar
            xc = [0.]*numcon

            y = [0.]*numcon
            suc = [0.]*numcon
            if (solsta == mosek.solsta.optimal):

                task.getxx(mosek.soltype.bas,  # Request the basic solution.
                           xx)
                task.getxc(mosek.soltype.bas, xc)
                task.gety(mosek.soltype.bas, y)
                task.getsuc(mosek.soltype.bas, suc)
                if print_detail:
                    print("Optimal solution: ")
                    for i in range(numvar):
                        print("x[" + str(i) + "]=" + str(xx[i]))
                    print("y")
                    for i in range(numcon):
                        print("y[" + str(i) + "]=" + str(y[i]))
                limit_force = xx[-1]
            else:
                if print_detail:
                    print("Other solution status")
                # return 0,[0.] * numvar
                limit_force = 0
        result["limit_force"] = limit_force
        result["contact_forces"] = xx[0:numvar-1]
        # dual optimization solutions

        # task.getsolutionslice(
        #     mosek.soltype.bas, mosek.solbasm.y, 0, len(elems)*3, y)

        result["xc"] = xc
        # normalize the displacement
        max_disp = 0
        for i in range(0, len(elems)):
            max_disp = max(max_disp, abs(y[i*3]), abs(y[i*3+1]))
        sum = 0
        element_index = 0
        for k, value in elems.items():
            sum += value.ll[0]*y[element_index*3]+value.ll[1] * \
                y[element_index*3+1]+value.ll[2]*y[element_index*3+2]
            element_index += 1
        if sum == 0:
            result["displacements"] = y[0:len(elems)*3]
        else:
            # y_scaled = np.array(y[0:len(elems)*3])/max_disp*np.sign(sum)
            # print(1/max_disp*np.sign(sum))
            # # #y_scaled[2::3] = y_scaled[2::3]*max_disp/np.sign(sum)
            # result["displacements"] = y_scaled.tolist()
            # result["displacements"] = (
            #     np.array(y[0:len(elems)*3])/sum).tolist()
            y_scaled = np.array(y[0:len(elems)*3])*np.sign(sum)
            result["displacements"] = y_scaled.tolist()

        result['suc'] = suc
    return result

