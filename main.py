import numpy as np
import pandas as pd
import math
from collections import deque
import itertools
import copy
import time
import sympy
from sympy import Symbol
from sympy import *
from sympy import simplify
from sympy import expand
from sympy import poly


#alpha = 2
#betta = 3
#gamma = 4
#sw = -1
#f = (a**(alpha - betta))**(-a**2 - a**-2)**gamma
#f = simplify(expand(f))
#print(f)
#f = simplify(f)
#print(f)
#print(f.coeff(a, -2))
#g = poly(2*a**2)
#dict = {p: f.coeff(a, p) for p in range(-100, 100) if f.coeff(a, p) != 0}
#print(dict)
#print(list(dict.keys()))

start_time = time.time()
#print("time elapsed: {:.2f}s".format(time.time() - start_time))
###############ТАНГЛЫ
def shift(list):
    vert_qual = int(len(list)/4)
    for v in range(0, vert_qual):
        slice = list[4*v:4*(v+1)]
        all_slice = []
        all_slice.append(slice)
        for i in range(0, 3):
            slice = slice[-1:] + slice[:-1]
            all_slice.append(slice)
        all_slice.sort()
        list[4*v:4*(v+1)] = all_slice[0]

    return list
def root_code(W, v_1, v_2, v_3, v_4, k):
    # W - матрица смежности (вершина-вершина) помеченной проекции (P,(v,e,f)) размера nx4 c записанными соседними вершинами в очередной строке
    # слева направо против часовой стрелки, начиная с любого соседа, W задается матрицей np.array
    # v_1 - номер помеченной вершины v
    # v_2 - номер вершины, соединенной с v ребром e
    # v_3 - номер вершины, соединенной с v ребром, не равном e и ограничивающем f
    # v_4 - номер вершины, соединенной с v_1, следующей после v_3 в направлении, задаваемом f
    # k - параметр, k=1 -если (v,e,f) задает обход против часовой стрелки, k=-1 -если по часовой

    n = W[:, 0].shape[0]  # узнаём число вершин степени 4
    # A = np.zeros((n, 4)) # подготавливаем матрицу A
    free = 2  # свободный номер вершины, его будем присваивать к вершине без номера
    Q = deque([v_1])  # очередь
    number = np.array([v_1])  # будет запоминать порядок вершин при нумерации = их новые номера, их индекс+1 равно новый номер
    incoming = W[v_1 - 1, :].copy()  # для удобства берем соседей очередной вершины, минус единица т.к. нумерация вершин с единицы, а элементов массива с нуля

    #########СОРТИРУЕМ incoming

    ## 1) Находим индекс v_2 в incoming (такие сложности вызваны возможностью того, что v_3 = 0 и программа не может отличить разные
    # граничные точки, т.к. все они имеют номер = 0

    index = np.where(incoming == v_2)[0]  # набор индексов из incoming, которым соответствуют значения, равные v_2

    len_index = index.shape[0]  # длина index
    #print('incoming')
    #print(incoming)
    #print(incoming.shape)
    #print(type(incoming))
    #print('W')
    #print(W)
    #print('v_1')
    #print(v_1)

    if len_index == 2:
        for i in index:
            if incoming[(i + k) % 4] == v_3:
                index_v_2 = i
    elif len_index == 3:
        for i in index:
            if (incoming[(i + k) % 4] == v_3) and (incoming[(i + 2 * k) % 4] == v_4):
                index_v_2 = i

    else:
        index_v_2 = index[0]

        ###2) Сортировка incoming
    incoming_sort = np.array([0, 0, 0, 0])

    for i in range(0, 4, 1):
        incoming_sort[i] = incoming[(index_v_2 + k * i) % 4]

        #########
    #print(incoming_sort)
        # str_number = 0 #будет говорить, какую строку матрицы A заполняем
        #########Теперь приступаем
    code = np.zeros((n * 4))
    m = 0
    end = 0
    while len(Q) != 0 and end == 0:
        Q.pop()
        #u = Q.pop()  # запоминаем последний добавленный элемент в очередь и удаляем его из очереди
        #print(u)
        #print(incoming)
        #print('number')
        #print(number)

        for w in incoming_sort:

            if w == 0:
                code[m] = 0
                m = m + 1
                #print(code)

            else:
                if np.isin(w, number) == False:
                    number = np.append(number, w)
                    free = free + 1
                    Q.appendleft(w)

                code[m] = np.where(number == w)[0][0] + 1
                m = m + 1
                #print(code)

        ##Сортируем incoming для следующего u.
        if len(Q) != 0:
            u = Q[-1]
        else: end = 1
        incoming = W[u - 1, :].copy()
        #print('Q')
        #print(Q)
        #print('incoming')
        #print(incoming)
        ###Если в incoming есть нули, нужно сортировать так, что бы вначале их было как можно больше
        if np.isin(0, incoming):
            index = np.where(incoming == 0)[0]  # набор индексов из incoming, которым соответствуют значения, равные 0
            len_index = index.shape[0]  # длина index

            if len_index == 2:
                q = 0
                for i in index:
                    if incoming[(i + k) % 4] == 0:
                        index_1 = i
                        q = 1
                    else:
                        c = 0
                        for x in number:
                            if incoming[(i + k) % 4] == x and c == 0 and q == 0:
                                c = 1
                                index_1 = i


            elif len_index == 3:
                for i in index:
                    if (incoming[(i + k) % 4] == 0) and (incoming[(i + 2 * k) % 4] == 0):
                        index_1 = i

            else:
                index_1 = index[0]
            ###

        else:
            check = 0
                ######Отлавливаем все индексы в incoming, где есть помеченная вершина с минимальным номером, входящая в incoming
            for tagged_node in number:
                if np.isin(tagged_node, incoming) and check == 0:
                    index = np.where(incoming == tagged_node)[0]
                    check = 1
                    tag_node = tagged_node

            ######

            len_index = index.shape[0]  # длина index
            if len_index == 2:
                for i in index:
                    if incoming[(i + k) % 4] == tag_node:
                        index_1 = i
            elif len_index == 3:
                for i in index:
                    if (incoming[(i + k) % 4] == tag_node) and (incoming[(i + 2 * k) % 4] == tag_node):
                        index_1 = i

            else:
                index_1 = index
            #print(len_index)
            ###2) Сортировка incoming
        for i in range(0, 4, 1):
            incoming_sort[i] = incoming[(index_1 + k * i) % 4]
            #print(incoming_sort[i])
        #print('incoming_sort')
        #print(incoming_sort)
        ####Преобразование code в матрицу A
    # A = np.reshape(code, (-1, 4))

    #print(code)
    # print(code.shape)
    # print(type(code))
    A = np.reshape(code, (-1, 4))
    code = list(code)

    return code
def find_face_code(W, v_1, v_2, v_3, v_4, k):
    # W - матрица смежности (вершина-вершина) помеченной проекции (P,(v,e,f)) размера nx4 c записанными соседними вершинами в очередной строке
    # слева направо против часовой стрелки, начиная с любого соседа, W задается матрицей np.array
    # v_1 - номер помеченной вершины v
    # v_2 - номер вершины, соединенной с v ребром e
    # v_3 - номер вершины, соединенной с v ребром, не равном e и ограничивающем f
    # v_4 - номер вершины, соединенной с v_1, следующей после v_3 в направлении, задаваемом f
    # k - параметр, k=1 -если (v,e,f) задает обход против часовой стрелки, k=-1 -если по часовой
    n = W[:, 0].shape[0]
    incoming = W[v_1 - 1, :].copy()
    index = np.where(incoming == v_2)[0]  # набор индексов из incoming, которым соответствуют значения, равные v_2

    len_index = index.shape[0]  # длина index
    if len_index == 2:
        for i in index:
            if incoming[(i + k) % 4] == v_3:
                index_v_2 = i
    elif len_index == 3:
        for i in index:
            if (incoming[(i + k) % 4] == v_3) and (incoming[(i + 2 * k) % 4] == v_4):
                index_v_2 = i
    else:
        index_v_2 = index[0]
    #print(index_v_2)
    def vertex_pass_1(W, u_first, ind):
        quality = 2
        u = u_first
        #print(u)

        v = W[u - 1, (ind + k) % 4]

        #print(v)
        if v == 0:
            return quality, u, ind + k
        while v != 0:
            quality += 1
            incoming_1 = W[v - 1, :].copy()
            for i in range(0, 4):
                if (incoming_1[i] == u and incoming_1[(i + k) % 4] != u):
                    index_1 = i

            u = v
            v = incoming_1[(index_1 + k) % 4]
        return quality, u, index_1 + k

    q, u, ind = vertex_pass_1(W, v_1, index_v_2)
    qual_check = 1
    qual_bound_cross = 0
    list = []
    list.append(q)
    for i in range(0, n):
        if 0 in W[i, :]:
            qual_bound_cross += np.where(W[i, :] == 0)[0].shape[0]
    while qual_check < qual_bound_cross:
        q, u, ind = vertex_pass_1(W, u, ind)
        qual_check += 1
        list.append(q)


    return list
def lexmin_face_code(W):
    n = W[:, 0].shape[0]
    c = 0
    list_1 = []
    list_2 = []
    for i in range(0, n):
        if 0 in W[i, :] and c != 1:
            incoming = W[i, :]
            v_1 = i + 1
            index = np.where(incoming == 0)[0][0]
            v_2 = incoming[(index) % 4]
            v_3 = incoming[(index + 1) % 4]
            v_4 = incoming[(index + 2) % 4]
            fc_1 = find_face_code(W, v_1, v_2, v_3, v_4, 1)
            v_2 = incoming[(index) % 4]
            v_3 = incoming[(index - 1) % 4]
            v_4 = incoming[(index - 2) % 4]
            fc_2 = find_face_code(W, v_1, v_2, v_3, v_4, -1)
            c = 1
    for i in range(0, len(fc_1)):
        list_1.append(fc_1[-i:] + fc_1[:-i])
        list_2.append(fc_2[-i:] + fc_2[:-i])
    list_1.sort()
    list_2.sort()
    return list_1[0], list_2[0]
def face_code_subset(W):
    lexmin_fc_1 = lexmin_face_code(W)[0]
    lexmin_fc_2 = lexmin_face_code(W)[1]
    n = W[:, 0].shape[0]
    canon_root = []
    canon_vertexes = []
    for i in range(0, n):
        if 0 in W[i, :]:
            v_1 = i + 1
            incoming = W[i, :]
            index = np.where(incoming == 0)[0]
            for ind in index:
                for k in [-1, 1]:
                    v_2 = incoming[(ind) % 4]
                    v_3 = incoming[(ind + k) % 4]
                    v_4 = incoming[(ind + 2*k) % 4]
                    if k == 1:
                        if find_face_code(W, v_1, v_2, v_3, v_4, k) == lexmin_fc_1:
                            canon_root.append([v_1, v_2, v_3, v_4, k])
                            canon_vertexes.append(v_1)
                    elif k == -1:
                        if find_face_code(W, v_1, v_2, v_3, v_4, k) == lexmin_fc_2:
                            canon_root.append([v_1, v_2, v_3, v_4, k])
                            canon_vertexes.append(v_1)
    return canon_root
def lexmin_root_code(W, v):
    v_1 = v
    incoming_v = W[v - 1, :].copy()
    all_root_code = []
    i = 0
    for direction in [-1, 1]:
        for e in range(0, 4, 1):
            k = direction
            v_2 = incoming_v[e]
            v_3 = incoming_v[(e + k) % 4]
            v_4 = incoming_v[(e + 2*k) % 4]
            #all_root_code[i] = root_code(W, v_1, v_2, v_3, v_4, k)
            all_root_code.append(root_code(W, v_1, v_2, v_3, v_4, k))
            i = i + 1
    #all_root_code = np.array(all_root_code)
    #print(all_root_code)
    #index_sort_arc = np.lexsort(np.rot90(all_root_code))
    #lexmin_rc = all_root_code[0]
    #lexmin_rc_matrix = np.reshape(lexmin_rc, (-1, 4))
    all_root_code.sort()
    return all_root_code[0]
def vertex_pass(W, u_first, v):
    u = u_first
    while (v != 0 and v != u_first):
        incoming_1 = W[v - 1, :].copy()

        for k in range(0, 4):
            if (incoming_1[k] == u and incoming_1[(k + 1) % 4] != u):
                index_1 = k
        u = v
        v = incoming_1[(index_1 + 1) % 4]
    if v == 0:
        return 1
    else:
        return 0
def find_cut_vertex(W):
    n = W[:, 0].shape[0]  # число вершин степени 4
    junc_point = []
    ##Ищем приграничные вершины
    bound_cross = []
    for i in range(0, n):
        if 0 in W[i, :]:
            bound_cross.append(i + 1)
    for u in bound_cross:
        incoming = W[u - 1, :].copy()
        index_zeros = np.where(incoming == 0)[0]
        not_zeros = np.delete(incoming, index_zeros)
        un_not_zeros = np.unique(not_zeros)
        if un_not_zeros.shape[0] == 2:
            for i in range(0, 4):
                if incoming[i] != 0 and incoming[(i+1) % 4] != 0 and incoming[i] != incoming[(i+1) % 4]:
                    v = incoming[(i+1) % 4]
                    if vertex_pass(W, u, v) == 1:
                        junc_point.append(u)

        elif un_not_zeros.shape[0] == 3:
            for i in range(0, 4):
                if incoming[i] == 0:
                    j = i
            if vertex_pass(W, u, incoming[(j+2) % 4]) == 1:
                junc_point.append(u)
            elif vertex_pass(W, u, incoming[(j+3) % 4]) == 1:
                junc_point.append(u)

        for i in range(0, 4):
            if incoming[i] == 0 and incoming[(i+1) % 4] != 0 and incoming[(i+2) % 4] == 0 and incoming[(i+3) % 4] != 0 and incoming[(i+3) % 4] != incoming[(i+1) % 4] and u not in junc_point:
                junc_point.append(u)
    return junc_point
def prev(W):
    if W.shape[0] == 1:

        return []
    else:
        n = W[:, 0].shape[0]  # число вершин степени 4
        jc = find_cut_vertex(W)
        bound_cross = []
        for i in range(0, n):
            if 0 in W[i, :]:
                bound_cross.append(i + 1)
    ####
        #bound_cross = np.array(bound_cross)
        not_cut_bc = [v for v in bound_cross if v not in jc] ##приграничные точки, не являющиеся точками сочленения

    ##поиск подходящей для удаления вершины (такой что root_code лексикограф. минимален)
        all_root_code = []
        for v in not_cut_bc:
            all_root_code.append(lexmin_root_code(W, v))

        #all_root_code = np.array(all_root_code)
        #index_sort_arc = np.lexsort(np.rot90(all_root_code))
        all_root_code.sort()
        lexmin_rc = all_root_code[0]
        #print(lexmin_rc)
        u_for_cuts = []
        for u in not_cut_bc:
            u_root_code = lexmin_root_code(W, u)
            #print(u_root_code)
            if u_root_code == lexmin_rc:
                u_for_cuts.append(u) ###нужная нам вершина дял удаления
        u_for_cut = u_for_cuts[0]
        #print('u_for_cuts')
        #print(u_for_cuts)
        #print('Удаленная вершина')
        #print(u_for_cut)
        ###удаление вершины u_for_cut


        #print('u_for_cut')
        #print(u_for_cut)
        #print('prev(R_vv)')
        #print(W_1)
        #print('--------')

        return u_for_cuts, lexmin_rc
def rcode(W):
    n = W[:, 0].shape[0]
    fc_subset = face_code_subset(W)
    list = []
    cut_vertex = find_cut_vertex(W)
    R_set = []
    canon_vertex = []
    for root in fc_subset:
        if root[0] not in cut_vertex:
            R_set.append(root)

    #print(R_set)
    for root in R_set:
        list.append(root_code(W, root[0], root[1], root[2], root[3], root[4]))
    list.sort()
    for root in R_set:
        if root_code(W, root[0], root[1], root[2], root[3], root[4]) == list[0]:
            canon_vertex.append(root[0])
    if len(list) != 0:
        return canon_vertex, list[0]
    else: return 0
def leg_numbers_func(T):
    leg_numbers = []  # leg_numbers[i] - номер ноги,куда приходит i-тая вершина
    max_e = 3
    first_elements = []
    second_elements = []
    third_elements = []
    for list in T:
        first_elements.append(list[0])
        second_elements.append(list[1])
        third_elements.append(list[2])
    ###первая вершина
    alpha = first_elements[0]
    m_1 = (second_elements[0]) % 4
    if alpha == - 1:
        if m_1 == 0:
            x = 0
        elif m_1 == 3:
            x = 2
        else:
            x = 1
        leg_numbers.append(m_1 + x * alpha)
    else:
        leg_numbers.append(m_1 + alpha)
    ###
    ##остальные вершины

    for i in range(1, len(T)):

        alpha = first_elements[i]
        m_1 = (leg_numbers[i - 1] + second_elements[i]) % third_elements[i - 1]
        if alpha == - 1:
            if m_1 == 0:
                x = 0
            elif m_1 == third_elements[i - 1] - 1:
                x = 2
            else:
                x = 1
            leg_numbers.append(m_1 + x * alpha)
        else:
            leg_numbers.append(m_1 + alpha)
    return leg_numbers
def e_list_func(T):
    e = [0, 1, 2, 3]  ##соседние грани
    e_list = []  # будем запоминать соседние грани на каждом шаге (внизу) (нулевой шаг- начальная вершина, второй шаг- первая вершина и т.д.
    e_list.append(e.copy())
    max_e = 3
    first_elements = []
    second_elements = []
    third_elements = []
    for list in T:
        first_elements.append(list[0])
        second_elements.append(list[1])
        third_elements.append(list[2])
    leg_numbers = leg_numbers_func(T)

    ##
    ########
    ##заполняем e_list:

    for i in range(0, len(T)):

        if i == 0:
            number = second_elements[0] % 4
            if first_elements[i] == -1:
                values_for_del = [e[number], e[(number + 1) % len(e)]]
                for val in values_for_del:
                    e.remove(val)
                if number == 0:
                    e.insert(0, e.pop())
            elif first_elements[i] == 0:
                e[(number + 1) % len(e)] = max_e + 1
                max_e += 1
            elif first_elements[i] == 1:
                e[number:(number + 1)] = [e[number], max_e + 1, max_e + 2]
                max_e = max_e + 2

        else:
            number = (leg_numbers[i - 1] + second_elements[i]) % third_elements[i - 1]
            if first_elements[i] == -1:
                values_for_del = [e[number], e[(number + 1) % len(e)]]
                for val in values_for_del:
                    e.remove(val)
                if number == 0:
                    e.insert(0, e.pop())
            elif first_elements[i] == 0:
                e[(number + 1) % len(e)] = max_e + 1
                max_e += 1
            elif first_elements[i] == 1:
                e[number: (number + 1)] = [e[number], max_e + 1, max_e + 2]
                max_e = max_e + 2
        e_list.append(e.copy())
    return e_list
def gluing_cross(T):
    ### Эта функция присоединяет вершину к T
    # T - каскадная диаграмма проекции T, у T[i] - первый элемент- альфы-итый, второй - m_i, третий - k_i (число ног на i-том уровне); формат T - список списков
    T_gl = []
    if len(T) == 0:
        for alpha in range(0, 2, 1):
            for m in range(0, 4, 1):
                T_1 = T.copy()
                T_1.append([alpha, m, 4 + 2 * alpha])
                T_gl.append(T_1)
    else:
        leg_qual = T[-1][2]
        if leg_qual >= 6:
            for m in range(0, leg_qual, 1):
                T_1 = T.copy()
                T_1.append([-1, m, leg_qual - 2])
                T_gl.append(T_1)
        for alpha in range(0, 2, 1):
            for m in range(0, leg_qual, 1):
                T_1 = T.copy()
                T_1.append([alpha, m, leg_qual + 2 * alpha])
                T_gl.append(T_1)



    return T_gl
def check_for_simple_tangl(T):
    e_list = e_list_func(T)
    once_neigh = []
    once_neigh_bnn = []
    once_neigh.append(set([0, 1]))
    once_neigh.append(set([1, 2]))
    once_neigh.append(set([2, 3]))
    once_neigh.append(set([3, 0]))
    for i in range(1, len(e_list)):
        e_now = e_list[i].copy()
        neigh = []
        for j in range(0, len(e_now) - 1):
            neigh.append(set((e_now[j:j + 2]).copy()))
        neigh.append(set([e_now[-1], e_now[0]]))
        #print(neigh)
        for ls in neigh:
            if ls in once_neigh and ls in once_neigh_bnn:
                return 0
        for ls in once_neigh:
            if ls not in neigh:
                once_neigh_bnn.append(ls)
        for ls in neigh:
            if ls not in once_neigh:
                once_neigh.append(ls)
        #print(once_neigh)
        #print(once_neigh_bnn)

    return 1
def cascade2vv(T):

    first_elements = []
    second_elements = []
    third_elements = []
    for list in T:
        first_elements.append(list[0])
        second_elements.append(list[1])
        third_elements.append(list[2])
    leg_numbers = leg_numbers_func(T)  # leg_numbers[i] - номер ноги,куда приходит i-тая вершина
    e_list = e_list_func(T)
    def way_up(rib):
        r = rib
        w = v - 1
        end = 0
        while end == 0 and w >= 0:
            ##соседи у w - 1
            first_e = e_list[w].copy()
            neigh = []
            for j in range(0, len(first_e) - 1):
                neigh.append((first_e[j:j + 2]).copy())
            neigh.append([first_e[-1], first_e[0]])
            ##
            if r not in neigh:
                end = 1
            else:
                w = w - 1
        return w
    def way_down(rib):
        r = rib
        end = 0  # проверка нахождения конца ребра
        w = v + 1  # претендент
        while end == 0 and w < len(T):
            first_e = e_list[w + 1].copy()
            neigh = []
            for j in range(0, len(first_e) - 1):
                neigh.append((first_e[j:j + 2]).copy())
            neigh.append([first_e[-1], first_e[0]])
            ##
            if r not in neigh:
                end = 1
            else:
                w += 1
        return w
    in_v = []
    v = -1
    r = [0, 1]
    in_v.append(way_down(r))
    r = [1, 2]
    in_v.append(way_down(r))
    r = [2, 3]
    in_v.append(way_down(r))
    r = [3, 0]
    in_v.append(way_down(r))
    VV = []
    VV.append(in_v.copy())


    for v in range(0, len(T)):
        alpha = first_elements[v]
        l = leg_numbers[v]
        in_v = []
        n = (leg_numbers[v - 1] + second_elements[v]) % third_elements[v - 1]
        if alpha == -1: #'кси'
            if v == len(T) - 1:
                in_v.append(len(T))
            else:
                r = [e_list[v + 1][l % len(e_list[v + 1])], e_list[v + 1][(l + 1) % len(e_list[v + 1])]]
                in_v.append(way_down(r))
            if v == 0:
                in_v = in_v + [-1, -1, -1]
            else:
                r = [e_list[v][(n + 1) % len(e_list[v])], e_list[v][(n + 2) % len(e_list[v])]]
                in_v.append(way_up(r))
                r = [e_list[v][(n) % len(e_list[v])], e_list[v][(n + 1) % len(e_list[v])]]
                in_v.append(way_up(r))
                r = [e_list[v][(n - 1) % len(e_list[v])], e_list[v][(n) % len(e_list[v])]]
                in_v.append(way_up(r))
        elif alpha == 0:
            if v == len(T) - 1:
                in_v = in_v + [len(T), len(T)]
            else:
                r = [e_list[v + 1][l], e_list[v + 1][(l + 1) % len(e_list[v + 1])]]
                in_v.append(way_down(r))
                r = [e_list[v + 1][(l + 1) % len(e_list[v + 1])], e_list[v + 1][(l + 2) % len(e_list[v + 1])]]
                in_v.append(way_down(r))
            if v == 0:
                in_v = in_v + [-1, -1]
            else:
                r = [e_list[v][(n + 1) % len(e_list[v])], e_list[v][(n + 2) % len(e_list[v])]]
                in_v.append(way_up(r))
                r = [e_list[v][(n) % len(e_list[v])], e_list[v][(n + 1) % len(e_list[v])]]
                in_v.append(way_up(r))
        elif alpha == 1:
            if v == len(T) - 1:
                in_v = in_v + [len(T), len(T), len(T)]
            else:
                r = [e_list[v + 1][l - 1], e_list[v + 1][(l) % len(e_list[v + 1])]]
                in_v.append(way_down(r))
                r = [e_list[v + 1][l], e_list[v + 1][(l + 1) % len(e_list[v + 1])]]
                in_v.append(way_down(r))
                r = [e_list[v + 1][l + 1], e_list[v + 1][(l + 2) % len(e_list[v + 1])]]
                in_v.append(way_down(r))
            if v == 0:
                in_v = in_v + [-1]
            else:
                r = [e_list[v][(n) % len(e_list[v])], e_list[v][(n + 1) % len(e_list[v])]]
                in_v.append(way_up(r))


        VV.append(in_v.copy())

    for i in range(0, len(VV)):
        for j in range(0, 4):
            if VV[i][j] == -1:
                VV[i][j] = 1
            elif VV[i][j] == len(T):
                VV[i][j] = 0
            else:
                VV[i][j] = VV[i][j] + 2
    VV = np.array(VV)
    return VV
def DFS(T, list=None, cascade_codes=None):
    if list is None:
        list = []
    if cascade_codes is None:
        cascade_codes = []
    if len(T) == 0:
        T_vv = np.array([[0, 0, 0, 0]])
    else:
        T_vv = cascade2vv(T)
    n = 6
    #print('T_vv')
    #print(T_vv)
    if len(T) == n - 1:
        cascade_codes.append(T)
    gl_cross = gluing_cross(T)
    if len(T) < n - 1:
        for R in gl_cross:
            #if len(R) == 6 and R[-1][2] == 8 and check_for_simple_1(R) == 0:
                #print(R)
            if check_for_simple_tangl(R) == 1:
                R_vv = cascade2vv(R)
                v = R_vv.shape[0]
                #print(R_vv)
                if v in prev(R_vv)[0]:
                    r = prev(R_vv)[1]
                    if r not in list:
                        #print(v)
                        DFS(R, list, cascade_codes)
                        list.append(r)



    return list, cascade_codes
###############


###############ПРОЕКЦИИ НА ТОР
def check_for_simple_project_1(T, star, r):
    #новая функция
    # T- каскадная диаграмма простого тангла
    # star - звезда [p, m, d]
    # r - номер ребра из последовательности (1, 2, ... ,2*k), соединяющегося с левой верхней параллелью из звезды (однозначно определяет склейку тангла T
    # и звезды star)

    p = star[0]
    m = star[1]
    d = star[2]
    k = p + m + d
    e_list = e_list_func(T)
    #
    once_neigh = []  # все, кто был соседями до последнего уровня каскадной диаграммы
    for edge in e_list:
        for i in range(0, len(edge) - 1):
            if set(edge[i:i + 2]) not in once_neigh:
                once_neigh.append(set(edge[i:i + 2]))
        once_neigh.append(set([edge[0],edge[-1]]))
    #
    exist_face = set(e_list[-2])  # существующие до последнего уровня грани
    #
    last_not_neigh = []
    for i in range(0, len(e_list[-1])):
        if i != 0 and i != (len(e_list[-1]) - 1):
            for j in [item for item in range(0, len(e_list[-1])) if item not in [i - 1, i, i + 1]]:
                last_not_neigh.append(set([e_list[-1][i], e_list[-1][j]]))
        elif i == 0:
            for j in range(2, len(e_list[-1]) - 1):
                last_not_neigh.append(set([e_list[-1][i], e_list[-1][j]]))
        elif i == (len(e_list[-1]) - 1):
            for j in range(1, len(e_list[-1]) - 3):
                last_not_neigh.append(set([e_list[-1][i], e_list[-1][j]]))
    #
    last_e = e_list[-1]
    #print('last_not_neigh')
    #print(last_e)
    #print(last_not_neigh)
    #print(once_neigh)
    applicants = [item_set for item_set in last_not_neigh if (item_set <= exist_face) and (item_set in once_neigh) ] # претенденты
    last_e = e_list[-1]
    ##соединение со звездой
    r = r - 1
    bind_ribs = [[(r + i) % (2 * k) + 1, (r + k + p - 1 - i) % (2 * k) + 1] for i in range(0, p)] + \
                [[(r + p + i) % (2 * k) + 1, (r + 2 * k - m - 1 - i) % (2 * k) + 1] for i in range(0, d)] + \
                [[(r + p + d + i) % (2 * k) + 1, (r + 2 * k - 1 - i) % (2 * k) + 1] for i in range(0, m)]
    bind_face = []
    for br in bind_ribs:
        first_rib = int(min(br)) - 1
        second_rib = int(max(br)) - 1
        #print([first_rib, second_rib])
        bf_1 = [last_e[first_rib], last_e[(second_rib + 1) % len(last_e)]]
        bf_2 = [last_e[(first_rib + 1) % len(last_e)], last_e[second_rib]]
        if set(bf_1) not in [set(item) for item in bind_face]:
            bind_face.append(bf_1)
        if bf_2 not in [set(item) for item in bind_face]:
            bind_face.append(bf_2)
    #print('last_e')
    #print(last_e)
    #print('bind_ribs')
    #print(bind_ribs)
    #print('bind_face')
    #print(bind_face)
    #print('applicants')
    #print(applicants)
    for app in applicants:
        app_1 = list(app)
        f_1 = app_1[0]
        f_2 = app_1[1]
        for bf_1 in bind_face:
            for bf_2 in bind_face:
                if f_1 in bf_1:
                    new_f_1 = [item for item in bf_1 if item != f_1][0]
                    if f_2 in bf_2:
                        new_f_2 = [item for item in bf_2 if item != f_2][0]
                        if set([new_f_1, new_f_2]) in applicants and set([new_f_1, new_f_2]) != app:
                            #print('app')
                            #print(app)
                            #print('new')
                            #print([f_1,f_2])
                            #print([new_f_1, new_f_2])
                            return 0


    return 1
def check_for_simple_project(T, star, r):
    #T- каскадная диаграмма простого тангла
    #star - звезда [p, m, d]
    #r - номер ребра из последовательности (1, 2, ... ,2*k), соединяющегося с левой верхней параллелью из звезды (однозначно определяет склейку тангла T
    # и звезды star)

    p = star[0]
    m = star[1]
    d = star[2]
    k = p + m + d
    e_list = e_list_func(T)
    #
    once_neigh = [] #все, кто был соседями до последнего уровня каскадной диаграммы
    for edge in e_list:
        for i in range(0, len(edge) - 1):
            if set(edge[i:i+2]) not in once_neigh:
                once_neigh.append(set(edge[i:i+2]))
        once_neigh.append(set(edge[-1], edge[0]))
    #
    exist_face = set(e_list[-2]) #существующие до последнего уровня грани
    #
    last_not_neigh = []
    for i in range(0, len(e_list[-1])):
        if i != 0 and i != (len(e_list[-1]) - 1):
            for j in [item for item in range(0, len(e_list[-1])) if item not in [i - 1, i, i + 1]]:
                last_not_neigh.append(set(e_list[-1][i], e_list[-1][j]))
        elif i == 0:
            for j in range(2, len(e_list[-1])):
                last_not_neigh.append(set(e_list[-1][i], e_list[-1][j]))
        elif i == (len(e_list[-1]) - 1):
            for j in range(0, len(e_list[-1]) - 2):
                last_not_neigh.append(set(e_list[-1][i], e_list[-1][j]))
    #
    last_e = e_list[-1]
    applicants = [] #претенденты
    last_e = e_list[-1]

    #######cтарое
    # for i in range(0, len(last_e)):
    #     for j in range(0, len(last_e)):
    #         j_1 = (j + i) % len(last_e)
    #         if int(abs((i - j_1))) % len(last_e) >= 2:
    #             if [last_e[i], last_e[j_1]] not in last_not_neigh and [last_e[j_1], last_e[i]] not in last_not_neigh:
    #                 last_not_neigh.append([last_e[i], last_e[j_1]])
    # for i in range(0, len(e_list) - 1):
    #     for face in e_list[i]:
    #         if face not in exist_face:
    #             exist_face.append(face)
    #     for j in range(0, len(e_list[i]) - 1):
    #         once_neigh.append((e_list[i][j:j + 2]).copy())
    #     once_neigh.append([e_list[i][-1], e_list[i][0]])
    # for ls in last_not_neigh:
    #     if ls[0] in exist_face and ls[1] in exist_face:
    #         if ls in once_neigh:
    #             applicants.append(ls)
    #######cтарое

    ##соединение со звездой
    r = r - 1
    bind_ribs = [[(r + i) % (2 * k) + 1, (r + k + p - 1 - i) % (2 * k) + 1] for i in range(0, p)] + \
                [[(r + p + i) % (2 * k) + 1, (r + 2 * k - m - 1 - i) % (2 * k) + 1] for i in range(0, d)] + \
                [[(r + p + d + i) % (2 * k) + 1, (r + 2 * k - 1 - i) % (2 * k) + 1] for i in range(0, m)]
    bind_face = []
    for br in bind_ribs:
        first_rib = int(min(br))
        second_rib = int(max(br))
        if [last_e[first_rib - 1], last_e[second_rib % len(last_e)]] not in bind_face and [last_e[second_rib % len(last_e)], last_e[first_rib - 1]] not in bind_face:
            bind_face.append([last_e[first_rib - 1], last_e[second_rib % len(last_e)]])
        if [last_e[first_rib % len(last_e)], last_e[(second_rib - 1)%len(last_e)]] not in bind_face and [last_e[(second_rib - 1)%len(last_e)], last_e[first_rib % len(last_e)]] not in bind_face:
            bind_face.append([last_e[first_rib % len(last_e)], last_e[(second_rib - 1)%len(last_e)]])
    for app in applicants:
        f_1 = app[0]
        f_2 = app[1]
        for bf in bind_face:
            if f_1 == bf[0]:
                new_f_1 = bf[1]
            elif f_1 == bf[1]:
                new_f_1 = bf[0]
            if f_2 == bf[0]:
                new_f_2 = bf[1]
            elif f_2 == bf[1]:
                new_f_2 = bf[0]
        if [new_f_1, new_f_2] in applicants or [new_f_2, new_f_1] in applicants:
            return 0
    return 1
def cascade2ve(T):
    if len(T) == 0:
        return [[1, 2, 3, 4]], 2
    VE = []
    first_elements = []
    second_elements = []
    third_elements = []
    for ls in T:
        first_elements.append(ls[0])
        second_elements.append(ls[1])
        third_elements.append(ls[2])
    leg_numbers = leg_numbers_func(T)  # leg_numbers[i] - номер ноги,куда приходит i-тая вершина
    e_list = e_list_func(T)
    edges = [] #нумеруем ребра, индекс равно номер ребра

    for ls in e_list[::-1]:
        for i in range(0, len(ls) - 1):
            if ls[i:i + 2] not in edges:
                edges.append(ls[i:i + 2])
        if [ls[-1], ls[0]] not in edges:
            edges.append([ls[-1], ls[0]])
    ##
    #v = -1
    in_v = []
    in_v.append(edges.index([0, 1]))
    in_v.append(edges.index([1, 2]))
    in_v.append(edges.index([2, 3]))
    in_v.append(edges.index([3, 0]))
    VE.append(in_v.copy())
    for v in range(0, len(T)):
        alpha = first_elements[v]
        l = leg_numbers[v]
        if v != 0:
            n = (leg_numbers[v - 1] + second_elements[v]) % third_elements[v - 1]
        else:
            n = second_elements[v] % 4
        in_v = []
        if alpha == -1:
            r = [e_list[v + 1][l], e_list[v + 1][(l + 1) % len(e_list[v + 1])]]
            in_v.append(edges.index(r))
            r = [e_list[v][(n + 1) % len(e_list[v])], e_list[v][(n + 2) % len(e_list[v])]]
            in_v.append(edges.index(r))
            r = [e_list[v][(n) % len(e_list[v])], e_list[v][(n + 1) % len(e_list[v])]]
            in_v.append(edges.index(r))
            r = [e_list[v][(n - 1) % len(e_list[v])], e_list[v][(n) % len(e_list[v])]]
            in_v.append(edges.index(r))
        elif alpha == 0:
            r = [e_list[v + 1][l], e_list[v + 1][(l + 1) % len(e_list[v + 1])]]
            in_v.append(edges.index(r))
            r = [e_list[v + 1][(l + 1) % len(e_list[v + 1])], e_list[v + 1][(l + 2) % len(e_list[v + 1])]]
            in_v.append(edges.index(r))
            r = [e_list[v][(n + 1) % len(e_list[v])], e_list[v][(n + 2) % len(e_list[v])]]
            in_v.append(edges.index(r))
            r = [e_list[v][(n) % len(e_list[v])], e_list[v][(n + 1) % len(e_list[v])]]
            in_v.append(edges.index(r))
        elif alpha == 1:
            r = [e_list[v + 1][(l - 1) % len(e_list[v + 1])], e_list[v + 1][(l) % len(e_list[v + 1])]]
            in_v.append(edges.index(r))
            r = [e_list[v + 1][(l) % len(e_list[v + 1])], e_list[v + 1][(l + 1) % len(e_list[v + 1])]]
            in_v.append(edges.index(r))
            r = [e_list[v + 1][(l + 1) % len(e_list[v + 1])], e_list[v + 1][(l + 2) % len(e_list[v + 1])]]
            in_v.append(edges.index(r))
            r = [e_list[v][(n) % len(e_list[v])], e_list[v][(n + 1) % len(e_list[v])]]
            in_v.append(edges.index(r))
        VE.append(in_v.copy())
    for i in range(0, len(VE)):
        for j in range(0, 4):
            VE[i][j] = VE[i][j] + 1
    #VE = list(reversed(VE))
    return VE, int(third_elements[-1]/2)

    ##возвращает массив массивов VE, VE[i] - список ребер, входящих в i+1 вершину, слева направо против часовой стрелки
def generation_stars(k):
    ##возвращает cписок списков (p, m, d), где m- кол-во меридиан, p- кол-во параллелей, d- кол-во диагоналей
    ##генерация всех неизоморфных троек (m, p, d)
    all_stars = []
    for p in range(1, k):
        for m in range(1, k - p + 1):
            if m <= p and (k - m - p) <= m:
                star = [p, m, k - m - p]
                if star not in all_stars:
                    all_stars.append(star)
    return all_stars
def tangle_join_star(VE, star, r, need=None):
    #возвращает список всех VE, полученных вращением тангла и присоединением его к звезде
    #VE- матрица VE тангла, VE- список списков
    #k- число ног / 2
    #star- список (p, m, d), где p- кол-во параллелей, m- кол-во меридиан, d- кол-во диагоналей
    # r - номер ребра из последовательности (1, 2, ... ,2*k), соединяющегося с левой верхней параллелью из звезды (однозначно определяет склейку тангла T
    # и звезды star)
    p = star[0]
    m = star[1]
    d = star[2]
    k = p + m + d
    r = r - 1
    bind_ribs = [[(r + i)%(2*k)+1, (r + k + p - 1 - i)%(2*k)+1] for i in range(0, p)] + \
                    [[(r + p + i)%(2*k)+1, (r + 2 * k - m - 1 - i)%(2*k)+1] for i in range(0, d)] + \
                    [[(r + p + d + i)%(2*k)+1, (r + 2 * k - 1 - i)%(2*k)+1] for i in range(0, m)]
    VE_1 = copy.deepcopy(VE)
    len_ve = len(VE)
    for i in range(0, len_ve):
        for j in range(0, 4):
            for br in bind_ribs:
                if VE_1[i][j] in br:
                    VE_1[i][j] = min(br)
    if need == None:
        return VE_1
    else:
        return VE_1, bind_ribs
#
def VE_rootcode(VE, v, j, k):
    #VE - матрица VE проекции
    #v - меченая вершина
    #r - меченое ребро, r = VE[v - 1][j]
    #k - меченая грань, k = 1 для грани против часовой от ребра (если выходить из v), k = -1 для грани по часовой
    def find_neigh(vert, rib_1):
        for i in range(0, len(VE)):
            if rib_1 in VE[i] and i != (vert - 1):
                return i + 1
        if VE[vert - 1].count(rib_1) == 2:
            return vert
        return 0
    all_edge = []
    for vert in VE:
        for r in vert:
            all_edge.append(r)
    all_edge = list(set(all_edge))
    number_edges = [0] * len(all_edge)
    r = VE[v - 1][j]
    code = []
    free = 2
    free_edge = 2
    number = [0] * len(VE)
    number[v - 1] = 1
    number_edges[all_edge.index(r)] = 1
    Q = deque([v])
    incoming = deque([r])
    while len(Q) != 0:
        u = Q.pop()
        rib = incoming.pop()
        if u == v:
            index = j
        else:
            index = VE[u - 1].index(rib)
        for i in range(0, 4):
            edge = VE[u - 1][(index + k * i) % 4]
            w = find_neigh(u, edge)
            id = all_edge.index(edge)
            if number_edges[id] == 0:
                number_edges[id] = free_edge
                free_edge += 1
            code.append(number_edges[id])
            if number[w - 1] == 0 and w != 0:
                number[w - 1] = free
                free = free + 1
                Q.appendleft(w)
                incoming.appendleft(VE[u - 1][(index + k * i) % 4])

    return shift(code)
def VE_lexmin_rootcode(VE, need = None):
    all_rootcode = []
    list_1 = []
    for i in range(0, len(VE)):
        v = i + 1
        for j in range(0, 4):
            for k in [-1, 1]:
                rc = VE_rootcode(VE, v, j, k)
                all_rootcode.append(rc)
                list_1.append((rc, v, j, k))
    all_rootcode.sort()
    if need == None:
        return all_rootcode[0]
    else:
        q = []
        for ls in list_1:
            if ls[0] == all_rootcode[0]:
                q.append((ls[1], ls[2], ls[3]))
        return all_rootcode[0], q
###подраздел: танглы с использованием ve-кодов (для увеличения скорости)
def VE_lexmin_rootcode_vert(VE, v):
    all_rootcode = []
    for j in range(0, 4):
        for k in [-1, 1]:
            all_rootcode.append(VE_rootcode(VE, v, j, k))
    all_rootcode.sort()
    return all_rootcode[0]
def find_border_vertex(VE):
    def find_neigh(vert, rib_1):
        for i in range(0, len(VE)):
            if rib_1 in VE[i] and i != (vert - 1):
                return i + 1
        if VE[vert - 1].count(rib_1) == 2:
            return vert
        return 0
    border_vert = []
    for i in range(0, len(VE)):
        check = 0
        for j in range(0, 4):
            if find_neigh(i + 1, VE[i][j]) == 0 and check == 0:
                check = 1
                border_vert.append(i + 1)
    return border_vert
def ve_prev(VE):
    def find_neigh(vert, rib_1):
        for i in range(0, len(VE)):
            if rib_1 in VE[i] and i != (vert - 1):
                return i + 1
        if VE[vert - 1].count(rib_1) == 2:
            return vert
        return 0
    def vert_pass(vert, rib):
        v = find_neigh(vert, rib)
        r = rib
        while v != 0 and v != vert:
            ind = VE[v - 1].index(r)
            r = VE[v - 1][(ind + 1) % 4]
            v = find_neigh(v, r)
        if v == 0:
            return 1
        return 0
    border_vert = find_border_vertex(VE)
    candidates = []
    zero_edges = []
    cut_vert = []
    for v in border_vert:
        zer_edg = []
        for j in range(0, 4):
            if find_neigh(v, VE[v - 1][j]) == 0:
                zer_edg.append(j)
        if len(zer_edg) < 3:
            candidates.append(v)
            zero_edges.append(zer_edg.copy())
    for i in range(0, len(candidates)):
        v = candidates[i]
        zer_edge = zero_edges[i]
        if len(zer_edge) == 2:
            if (zer_edge[1] - zer_edge[0]) % 2 == 0:
                cut_vert.append(v)
            else:
                index = zer_edge[1]
                if find_neigh(v, VE[v - 1][(index + 1)%4]) != find_neigh(v, VE[v - 1][(index + 2)%4]):
                    if vert_pass(v, VE[v - 1][(index + 2)%4]) == 1:
                        cut_vert.append(v)
        elif len(zer_edge) == 1:
            index = zer_edge[0]
            not_zeros_neigh = []
            for j in range(1, 4):
                not_zeros_neigh.append(find_neigh(v, VE[v - 1][(index + j) % 4]))
            if not_zeros_neigh[0] == not_zeros_neigh[1]:
                if vert_pass(v, VE[v - 1][(index + 3) % 4]) == 1:
                    cut_vert.append(v)
            elif not_zeros_neigh[1] == not_zeros_neigh[2]:
                if vert_pass(v, VE[v - 1][(index + 2) % 4]) == 1:
                    cut_vert.append(v)
            elif vert_pass(v, VE[v - 1][(index + 2) % 4]) == 1 or vert_pass(v, VE[v - 1][(index + 3) % 4]) == 1:
                cut_vert.append(v)
    ##
    all_root_codes = []
    canon_vert = []
    app_vert = [v for v in border_vert if v not in cut_vert]
    for v in app_vert:
        all_root_codes.append(VE_lexmin_rootcode_vert(VE, v))
    sorted_all_root_codes = sorted(all_root_codes)
    for i in range(0, len(app_vert)):
        if all_root_codes[i] == sorted_all_root_codes[0]:
            canon_vert.append(app_vert[i])

    return canon_vert, sorted_all_root_codes[0]
def DFS_ve(T, n, list = None, cascade_codes = None):
    if cascade_codes == None:
        cascade_codes = []
    if list is None:
        list = []
    if len(T) == 0:
        T_ve = [[1, 2, 3, 4]]
    else:
        T_ve = cascade2ve(T)[0]
    #n = 2
    if len(T) == n - 1:
        cascade_codes.append(T)
    gl_cross = gluing_cross(T)
    if len(T) < n - 1:
        for R in gl_cross:
            if check_for_simple_tangl(R) == 1:
                R_ve = cascade2ve(R)[0]
                v = len(R_ve)
                prev = ve_prev(R_ve)
                if v in prev[0]:
                    r = prev[1]
                    if r not in list:
                        DFS_ve(R, n, list, cascade_codes)
                        list.append(r.copy())
    return list, cascade_codes

###
##проверка на появление двуугольных граней при приклеивании
def check_for_double_edges(T, star, r):
    #возвращает ноль, если двуугольных граней не появляется, и 1- иначе
    ve_tangl, k = cascade2ve(T)
    p = star[0]
    m = star[1]
    d = star[2]
    r = r - 1
    bind_ribs = [[(r + i)%(2*k)+1, (r + k + p - 1 - i)%(2*k)+1] for i in range(0, p)] + \
                    [[(r + p + i)%(2*k)+1, (r + 2 * k - m - 1 - i)%(2*k)+1] for i in range(0, d)] + \
                    [[(r + p + d + i)%(2*k)+1, (r + 2 * k - 1 - i)%(2*k)+1] for i in range(0, m)]
    bind_ribs_p = bind_ribs[0: p]
    bind_ribs_d = bind_ribs[p: p + d]
    bind_ribs_m = bind_ribs[p + d:]
    if p >= 2:
        for i in range(0, p - 1):
            br = bind_ribs_p[i: i + 2]
            check_1 = 0
            check_2 = 0
            for v in ve_tangl:
                if set([br[0][0], br[1][0]]) <= set(v):
                    check_1 = 1
                if set([br[0][1], br[1][1]]) <= set(v):
                    check_2 = 1
            if check_1 == 1 and check_2 == 1:
                return 1
    if d >= 2:
        for i in range(0, d - 1):
            br = bind_ribs_d[i: i + 2]
            check_1 = 0
            check_2 = 0
            for v in ve_tangl:
                if set([br[0][0], br[1][0]]) <= set(v):
                    check_1 = 1
                if set([br[0][1], br[1][1]]) <= set(v):
                    check_2 = 1
            if check_1 == 1 and check_2 == 1:
                return 1
    if m >= 2:
        for i in range(0, m - 1):
            br = bind_ribs_m[i: i + 2]
            check_1 = 0
            check_2 = 0
            for v in ve_tangl:
                if set([br[0][0], br[1][0]]) <= set(v):
                    check_1 = 1
                if set([br[0][1], br[1][1]]) <= set(v):
                    check_2 = 1
            if check_1 == 1 and check_2 == 1:
                return 1
    return 0

##приклеивание
def all_projects(cascade_codes_tangles):
    def reverse(VE):
        VE_rev = []
        for v in VE:
            VE_rev.append(list(reversed(v)))
        return VE_rev
    list_1 = []
    simple_projects = []
    all_root_codes = []
    all_ve_codes = []
    for tangl in cascade_codes_tangles:
        ve_tangl = cascade2ve(tangl)[0]
        if len(ve_tangl) > 1:
            k = int(tangl[-1][-1]/2)
        else: k = 4
        relevant_stars = generation_stars(k)
        for star in relevant_stars:
            for r in range(1, k + 1):
                if check_for_simple_project_1(tangl, star, r) == 1:
                    projection = tangle_join_star(ve_tangl, star, r)
                    check = 1
                    for vert in projection:
                        check_array = []
                        for edges in vert:
                            if edges not in check_array:
                                check_array.append(edges)
                        if len(check_array) < 4:
                            check = 0
                    rcode_of_projection = VE_lexmin_rootcode(projection)
                    if rcode_of_projection not in all_root_codes and check == 1:
                        all_root_codes.append(rcode_of_projection)
                        all_ve_codes.append(projection)
                        list_1.append([ve_tangl, star, r])
    #print('proverka')
    #print(VE_lexmin_rootcode(ve_test) in all_root_codes)
    #print('proverka')
    return list_1, all_ve_codes
def reverse(VE):
    VE_rev = []
    for v in VE:
        VE_rev.append(list(reversed(v)))
    return VE_rev
#casc_codes = DFS_ve([], 2)[-1]
#print(len(casc_codes))
#all_pr, all_ve = all_projects(casc_codes)

def wtf(list, num):
    f = open('C:\\Users\\Lenovo\\Desktop\\Диплом\\Защита\\Результаты для защиты\\five_vert\\' + str(num) + '.txt', 'a')
    for ve in list:
        f.write('[')
        for i in range(0, len(ve) - 1):
            vert = ve[i]
            for i_1 in range(0, 3):
                f.write(str(vert[i_1]))
                f.write(' ')
            f.write(str(vert[3]))
            f.write(';')
        vert = ve[-1]
        for j in range(0, 3):
            f.write(str(vert[j]))
            f.write(' ')
        f.write(str(vert[3]))
        f.write('],...')
        f.write('\n')
    f.close()
#wtf()
#for ve in all_ve:
    #print(VE_lexmin_rootcode(pr_t) == VE_lexmin_rootcode(ve))
#print('всего проекций')
#print(len(all_pr))
#j = 0
#for i in range(0, len(all_ve)):
    #pr = all_pr[i]
    #ve = all_ve[i]
    #ve_rev = reverse(ve)
    #rc, marked = VE_lexmin_rootcode(ve, 1)
    #if VE_lexmin_rootcode(ve) == VE_rootcode(ve_rev, marked[0], 3 - marked[1], marked[2]):
        #j += 1
        #print('оригинал')
        #print(pr)
        #print('отражение')
        #print([reverse(pr[0]), pr[1], pr[2]])
        #print('метка')
        #print(marked)
        #print('руткод')
        #print(rc)
#print(j)
#print("time elapsed: {:.2f}s".format(time.time() - start_time))
#for pr in all_pr:
    #print(pr)
    #print([cascade2ve(pr[0])[0], pr[1], pr[2]])
##
###############
###########################ЗАЦЕПЛЕНИЯ
def minimizing_vectors(list_vectors, ind=None):
    if ind == None:
        ind = 0
    #ind - индекс вектора из list_vectors, считающегося "каноническим"
    canon_vect = list_vectors[ind]
    #list_vectors- список векторов, для которых решается задача минимизации. Вектор = список [m, n]
    initial_vectors = []
    for vs in list_vectors:
        initial_vectors.append(np.array([vs]).T)
    q_initial = sum([np.dot(x.T, x)[0] for x in initial_vectors])
    M_plus = np.array([[1, 1], [0, 1]])
    M_minus = np.array([[1, -1], [0, 1]])
    L_plus = np.array([[1, 0], [1, 1]])
    L_minus = np.array([[1, 0], [-1, 1]])
    twistings = [M_plus, M_minus, L_plus, L_minus]
    #применение оператора matrix к векторам
    def operator(vectors, matrix):
        #vectors список векторов (столбцов np.array)
        # matrix - матрица преобразования
        result = []
        for vector in vectors:
            result.append(np.dot(matrix, vector))
        return result
    check = 0
    for twist in twistings:
        if check == 0:
            final_vectors = operator(initial_vectors, twist)
            q_final = sum([np.dot(x.T, x)[0] for x in final_vectors])
            if q_final < q_initial:
                check = 1
                F = twist.copy()
    if check == 0:
        return tuple([tuple(vs) for vs in list_vectors])

    while q_final < q_initial:
        initial_vectors = copy.deepcopy(final_vectors)
        q_initial = q_final
        check = 0
        for twist in twistings:
            if check == 0:
                final_vectors = operator(initial_vectors, twist)
                q_final = sum([np.dot(x.T, x)[0] for x in final_vectors])
                if q_final < q_initial:
                    F = np.dot(twist, F).copy()
                    check = 1
    rotation = np.array([[0, -1], [1, 0]])
    candidates = [F, np.dot(rotation, F), np.dot(np.dot(rotation, rotation), F), np.dot(np.dot(np.dot(rotation, rotation), rotation), F)]

    return tuple([tuple(np.dot(F, np.array([vs]).T).T[0]) for vs in list_vectors])
    ######################################################################################
    check = 0
    result = []
    for twist in candidates:
        # x_1 = np.dot(twist, np.array([[1], [0]]))[0, 0]
        # y_1 = np.dot(twist, np.array([[1], [0]]))[1, 0]
        # x_2 = np.dot(twist, np.array([[0], [1]]))[0, 0]
        # y_2 = np.dot(twist, np.array([[0], [1]]))[1, 0]
        canon_vect_1 = np.dot(twist, np.array(np.array([canon_vect]).T))
        if (canon_vect_1 >= 0).all() and check == 0:
            check == 1
            canon_operator = twist.copy()
    return (tuple(np.dot(canon_operator, np.array([vs]).T).T[0]) for vs in list_vectors)
            #return [list(np.dot(twist, np.array([vs]).T).T[0]) for vs in list_vectors]
            #return candidates
    #######################################################################
#проверка на эквивалентность орбит двух наборов векторов
def check_fe_canon_vect(list_1, list_2):
    #возвращает ноль, если два набора не эквивалентны, иначе возвращает список индексов элементов list_1 в list_2
    rotation = np.array([[0, -1], [1, 0]])
    twists = [np.array([[1, 0], [0, 1]]), rotation, np.dot(rotation, rotation), np.dot(np.dot(rotation, rotation), rotation)]
    for tw in twists:
        rot_list_1 = tuple([tuple(np.dot(tw, np.array([vs]).T).T[0]) for vs in list_1])
        if set(rot_list_1) == set(list_2):
            #print(tw)
            return tuple([list_2.index(vs) for vs in rot_list_1])
    return 0
#
#генерация всевозможных зацеплений из проекции, заданной ve-кодом или представлением тангл + звезда [tangl, star, r]
###инварианты зацеплений
#компоненты зацепления
def components(meshing):
    VE = meshing
    def find_neigh(vert, rib_1):
        for i in range(0, len(VE)):
            if rib_1 in VE[i] and i != (vert - 1):
                return i + 1
        if VE[vert - 1].count(rib_1) == 2:
            return vert
        return 0
    components = []
    edges = []
    for v in VE:
        edges = edges + v
    edges = list(set(edges))
    tagged_edges = []
    free_edges = edges.copy()
    ###

    ###
    while len(free_edges) != 0:
        component = []
        free_edge = free_edges[0]
        component.append(free_edge)
        check = 0
        for i in range(0, len(VE)):
            if free_edge in VE[i] and check == 0:
                check = 1
                r = VE[i][(VE[i].index(free_edge) + 2) % 4]
                j = find_neigh(i + 1, r) - 1
                #component.append(r)
                while r != free_edge:
                    component.append(r)
                    r = VE[j][(VE[j].index(r) + 2) % 4]
                    j = find_neigh(j + 1, r) - 1
                    #component.append(r)
        tagged_edges = tagged_edges + component
        components.append(component.copy())
        free_edges = [edge for edge in edges if edge not in tagged_edges]

    return components
def generation_junc(VE, ls):
    #функция генерирует все зацепления из заданной проекции
    #VE - руткод проекции
    #ls = [ve_tangl, star, r] - представление тангл + звезда
    def find_neigh(vert, rib_1):
        for i in range(0, len(VE)):
            if rib_1 in VE[i] and i != (vert - 1):
                return i + 1
        if VE[vert - 1].count(rib_1) == 2:
            return vert
        return 0
    def check_for_not_feature(ve_tangl, ve_mesh):
        # проверка на отсутствие "висячих" компонент зацепления и на отсуствие мест, где можно применить третье преобразование Рейдермейстера
        # аргументом является представление тангл+звезда
        def find_neigh(VE, vert, rib_1):
            for i in range(0, len(VE)):
                if rib_1 in VE[i] and i != (vert - 1):
                    return i + 1
            if VE[vert - 1].count(rib_1) == 2:
                return vert
            return 0
        #ve_tangl = mesh[0]
        #ve_mesh = tangle_join_star(ve_tangl, mesh[1], mesh[2])
        ###третье преобразование Рейдермейстера
        for i in range(0, len(ve_tangl)):
            v_1 = i + 1
            incoming = ve_tangl[i]
            for j in range(0, 4):
                r = incoming[j]
                w_1 = find_neigh(ve_tangl, v_1, r)
                w_2 = find_neigh(ve_tangl, v_1, incoming[(j + 1) % 4])
                if w_1 == w_2 and w_1 != 0:
                    incoming_1 = ve_tangl[w_1 - 1]
                    q_1 = incoming.index(r)
                    q_2 = incoming_1.index(r)
                    if q_1 in [0, 2] and q_2 in [0, 2]:
                        return 0
                    if q_1 in [1, 3] and q_2 in [1, 3]:
                        return 0
        ###
        ###наличие "висячих" компонент:
        cts = components(ve_mesh)
        for ct in cts:
            check_list = []
            for v in ve_mesh:
                ct_in_v = [vert for vert in v if vert in ct]
                if len(ct_in_v) != 0:
                    for r in ct_in_v:
                        if v.index(r) in [0, 2]:
                            check_list.append(1)
                        else:
                            check_list.append(0)
            if len(set(check_list)) == 1:
                return 0
        ###
        return 1
    #ls[0] = cascade2ve(ls[0])[0]
    tangl = ls[0]
    all_meshing = []
    all_meshing_tangl = []
    n = len(VE)
    all_placement = [list(item) for item in itertools.product([0, 1], repeat=n)]
    accel_place = []
    for item in all_placement:
        if [(x + 1) % 2 for x in item] not in accel_place:
            accel_place.append(item)
    for pl in accel_place:
        tangl = [ls[0][i][-pl[i]:] + ls[0][i][:-pl[i]] for i in range(0, len(VE))]
        ve_meshing = [VE[i][-pl[i]:] + VE[i][:-pl[i]] for i in range(0, len(VE))]
        if check_for_not_feature(tangl, ve_meshing) == 1:
            all_meshing.append(ve_meshing)
            all_meshing_tangl.append([tangl, ls[1], ls[2]])

    #for mesh in all_meshing:
        #print(VE_lexmin_rootcode(ve_test) == VE_lexmin_rootcode(mesh))
    return all_meshing_tangl, all_meshing
#
#генерация всевозможных зацеплений с заданным числом прекрестков
###инварианты зацеплений
def generation_meshing(n):
    all_meshing = []
    all_meshing_tangl = []
    all_tangl_pr, all_pr = all_projects(DFS_ve([], n)[-1])
    i = 0
    for ve in all_pr:
        g_junc_tangl_pr, gjunc = generation_junc(ve, all_tangl_pr[i])
        all_meshing = all_meshing + [item for item in gjunc]
        all_meshing_tangl = all_meshing_tangl + [item for item in g_junc_tangl_pr]
        i += 1
    #print(VE_lexmin_rootcode(ve_test) in [VE_lexmin_rootcode(tangle_join_star(ve_1[0], ve_1[1], ve_1[2])) for ve_1 in all_meshing_tangl])
    return all_meshing_tangl, all_meshing
#meshings = generation_meshing(3)[0]
#
###инварианты зацеплений
#Полином Кауфмана от представления [ve_tangl, star, r]
def kauffman_polynomial(ls):
    a = Symbol('a')
    def index_lexmin_poly(all_cond_poly):
        all_deg = []
        all_coeff = []
        for pol in all_cond_poly:
            dict = {p: pol.coeff(a, p) for p in range(-100, 100) if pol.coeff(a, p) != 0}
            degrees = list(dict.keys())
            coeff = []
            for key in degrees:
                coeff.append(dict[key])
            all_deg.append(degrees.copy())
            all_coeff.append(coeff.copy())
        all_deg_sorted = sorted(all_deg)
        index_lexmin_degs = [i for i in range(0, len(all_deg)) if all_deg[i] == all_deg_sorted[0]]
        if len(index_lexmin_degs) > 1:
            app_coeff = [all_coeff[i] for i in index_lexmin_degs]
            app_coeff_sorted = sorted(app_coeff)
            for i in index_lexmin_degs:
                if all_coeff[i] == app_coeff_sorted[0]:
                    return i
        else: return index_lexmin_degs[0]
    def sw(meshing):
        epsilon = []
        for ct in cts:
            e = 0
            for v in meshing:
                if set(v) <= set(ct):
                    ind_up = ct.index(v[0])
                    ind_down = ct.index(v[1])
                    if v[2] == ct[(ind_up + 1) % len(ct)]:
                        if v[3] == ct[(ind_down + 1) % len(ct)]:
                            e = e + 1
                        else: e = e - 1
                    elif v[2] == ct[(ind_up - 1) % len(ct)]:
                        if v[3] == ct[(ind_down + 1) % len(ct)]:
                            e = e - 1
                        else: e = e + 1
            epsilon.append(e)
        return sum(epsilon)
    #для n
    def find_d_sgn_n(d_1, segments):
        sgn = 0
        for segm in segments:
            check = 0
            if segm[0] == d_1:
                check = 1
                sgn += 1
                r_1 = segm[1]
                r_2 = segm[0]
                if (r_1 in up_d and up_d.index(r_1) > up_d.index(r_2)) or r_1 in right:
                    sgn += -1
                else: sgn += 1
            if segm[1] == d_1:
                check = 1
                sgn += 1
                r_2 = segm[1]
                r_1 = segm[0]
                if (r_1 in up_d and up_d.index(r_1) > up_d.index(r_2)) or r_1 in right:
                    sgn += -1
                else: sgn += 1
            if check == 1:
                return sgn
    def find_m_sgn(m_1, segments):
        sgn = 0
        m_1_down = down[up_m.index(m_1)]
        for segm in segments:
            if m_1 in segm:
                #l_1- входит вверх
                r_1 = segm[segm.index(m_1)]
                r_2 = [sg for sg in segm if sg != m_1][0]
            if m_1_down in segm:
                #l_2- входит вниз
                v_1 = segm[segm.index(m_1_down)]
                v_2 = [sg for sg in segm if sg != m_1_down][0]
        if v_1 == r_2 and v_2 == r_1:
            return 2
        #обработка l_1
        if r_2 in right or r_2 in up_d:
            sgn += -1
        elif r_2 in left_p or r_2 in left_d:
            sgn += 1
        elif r_2 in up_m:
            if up_m.index(r_2) > up_m.index(r_1):
                sgn += - 1
            else: sgn += 1
        elif r_2 in down:
            if down.index(r_2) > up_m.index(r_1):
                sgn += -1
            else: sgn += 1
        #обработка l_2
        if v_2 in right or v_2 in up_d:
            sgn += 1
        elif v_2 in left_p or v_2 in left_d:
            sgn += -1
        elif v_2 in up_m:
            if up_m.index(v_2) > down.index(v_1):
                sgn += 1
            else: sgn += -1
        elif v_2 in down:
            if down.index(v_2) > down.index(v_1):
                sgn += 1
            else:
                sgn += -1
        return sgn
    #для m
    def find_d_sgn_m(d_1, segments):
        sgn = 0
        for segm in segments:
            check = 0
            if segm[0] == d_1:
                check = 1
                sgn += -1
                r_1 = segm[1]
                r_2 = segm[0]
                if (r_1 in left_d and left_d.index(r_1) > left_d.index(r_2)) or r_1 in left_p + up_m + up_d + right:
                    sgn += -1
                else: sgn += 1
            if segm[1] == d_1:
                check = 1
                sgn += -1
                r_2 = segm[1]
                r_1 = segm[0]
                if (r_1 in left_d and left_d.index(r_1) > left_d.index(r_2)) or r_1 in left_p + up_m + up_d + right:
                    sgn += -1
                else: sgn += 1
            if check == 1:
                return sgn
    def find_p_sgn(p_1, segments):
        sgn = 0
        p_1_right = right[left_p.index(p_1)]
        for segm in segments:
            if p_1 in segm:
                #l_1- входит влево
                r_1 = segm[segm.index(p_1)]
                r_2 = [sg for sg in segm if sg != p_1][0]
            if p_1_right in segm:
                #l_2- входит вправо
                v_1 = segm[segm.index(p_1_right)]
                v_2 = [sg for sg in segm if sg != p_1_right][0]
        if v_1 == r_2 and v_2 == r_1:
            return 2
        #обработка l_1
        if r_2 in left_d + down:
            sgn += 1
        elif r_2 in up_m + up_d:
            sgn += -1
        elif r_2 in left_p:
            if left_p.index(r_2) < left_p.index(r_1):
                sgn += -1
            else: sgn += 1
        elif r_2 in right:
            if right.index(r_2) < left_p.index(r_1):
                sgn += -1
            else: sgn += 1
        #обработка l_2
        if v_2 in left_d + down:
            sgn += -1
        elif v_2 in up_m + up_d:
            sgn += 1
        elif v_2 in left_p:
            if left_p.index(v_2) < right.index(v_1):
                sgn += 1
            else: sgn += -1
        elif v_2 in right:
            if right.index(v_2) < right.index(v_1):
                sgn += 1
            else: sgn += -1
        return sgn
    #
    def closing_1(segments):
        j = 0
        while j < len(segments):
            changes = True
            while changes:
                changes = False
                for segm_1 in segments:
                    for segm_2 in [item for item in segments if item != segm_1]:
                        check = 0
                        for rib_1 in segm_1:
                            for rib_2 in segm_2:
                                if ([rib_1, rib_2] in bind_ribs or [rib_2, rib_1] in bind_ribs) and check == 0:
                                    check = 1
                                    changes = True
                                    segm_1 += segm_2
                                    segments.remove(segm_2)
            j += 1
        return segments
    def closing(segments):
        busy_ribs = []
        clos = []
        edges = left + down
        for edge in edges:
            if edge not in busy_ribs:
                busy_ribs.append(edge)
                winding = []
                for i in range(0, len(segments)):
                    #print(segments)
                    #print(i)
                    if segments[i][-1] == edge:
                        winding += [segments[i][-1], segments[i][0]]
                        if segments[i][0] in edges:
                            busy_ribs.append(segments[i][0])
                    elif segments[i][0] == edge:
                        winding += [segments[i][0], segments[i][-1]]
                        if segments[i][-1] in edges:
                            busy_ribs.append(segments[i][-1])
                while set([winding[0], winding[-1]]) not in [set(item) for item in bind_ribs]:
                    last_edge = winding[-1]
                    for br in bind_ribs:
                        if last_edge == br[0]:
                            last_edge_1 = br[1]
                        elif last_edge == br[1]:
                            last_edge_1 = br[0]
                    for segm in segments:
                        if segm[-1] == last_edge_1:
                            winding += [segm[-1], segm[0]]
                            if segm[0] in edges:
                                busy_ribs.append(segm[0])
                            if segm[1] in edges:
                                busy_ribs.append(segm[1])
                        elif segm[0] == last_edge_1:
                            winding += [segm[0], segm[-1]]
                            if segm[-1] in edges:
                                busy_ribs.append(segm[-1])
                            if segm[0] in edges:
                                busy_ribs.append(segm[0])
                clos.append(winding.copy())
        return clos
    def coords(edge):
        left_1 = left.copy()
        up_1 = up.copy()
        right_1 = right.copy()
        down_1 = down.copy()
        left_1.reverse()
        right_1.reverse()
        if edge in left_1:
            i = left_1.index(edge)
            return [0, (i + 1)/(len(left_1) + 1)]
        elif edge in down_1:
            i = down_1.index(edge)
            return [(i + 1)/(len(up_1) + 1), 0]
        elif edge in right_1:
            i = right_1.index(edge)
            return [1, (i + 1 + len(left_d))/(len(left_1) + 1)]
        elif edge in up_1:
            i = up_1.index(edge)
            return [(i + 1)/(len(up_1) + 1), 1]
    def gcd(a_1, b_1):
        if a_1 == 0 and b_1 == 0:
            return 1
        else: return math.gcd(a_1, b_1)
    ve_tangl = ls[0]
    n = len(ve_tangl)
    star = ls[1]
    p = star[0]
    m = star[1]
    d = star[2]
    r = ls[2]
    k = int((p + m + d)/2)
    meshing, bind_ribs = tangle_join_star(ve_tangl, star, r, 1)
    #print('ve')
    #print(meshing)
    cts = components(meshing)
    #print('components')
    #print(cts)
    sort_edges = list(range(1, 2 * k + 1))
    sort_edges = sort_edges[r - 1:] + sort_edges[:r - 1]
    #sort_bind_ribs = [[sort_edges.index(item[0]), sort_edges.index(item[1])] for item in bind_ribs]
    left_p = [br[0] for br in bind_ribs[0:p]]
    left_d = [br[0] for br in bind_ribs[p:p + d]]
    left = left_p + left_d
    right = [br[1] for br in bind_ribs[0:p]]
    down = [br[0] for br in bind_ribs[p + d:]]
    up_m = [br[1] for br in bind_ribs[p + d:]]
    up_d = [br[1] for br in bind_ribs[p:p + d]]
    up = up_m + up_d
    #print('bind_ribs')
    #print(bind_ribs)
    border_ribs = []
    for pair in bind_ribs:
        if pair[0] not in border_ribs:
            border_ribs.append(pair[0])
        if pair[1] not in border_ribs:
            border_ribs.append(pair[1])
    all_condition = [list(item) for item in itertools.product([0, 1], repeat=n)]
    #print(all_condition)
    #all_condition = [[1, 1, 1]]
    sw = sw(meshing)
    all_cond_poly = []
    all_windings = []
    for condition in all_condition:
        segments_cond_tangl = []
        alpha = 0
        beta = 0
        gamma = 0
        for number in condition:
            if number == 0:
                alpha += 1
            else:
                beta += 1
        for i in range(0, len(ve_tangl)):
            v = ve_tangl[i]
            if condition[i] == 0:
                line_1 = [v[0], v[3]]
                line_2 = [v[1], v[2]]
            else:
                line_1 = [v[0], v[1]]
                line_2 = [v[2], v[3]]
            if len(segments_cond_tangl) != 0:
                check_1 = 0
                check_2 = 0
                for curv in segments_cond_tangl:
                ####
                    if line_1[0] in curv:
                        if line_1[1] not in curv:
                            check_1 = 1
                            curv.append(line_1[1])
                    elif line_1[1] in curv:
                        if line_1[0] not in curv:
                            check_1 = 1
                            curv.append(line_1[0])
                ####
                    if line_2[0] in curv:
                        if line_2[1] not in curv:
                            check_2 = 1
                            curv.append(line_2[1])
                    elif line_2[1] in curv:
                        if line_2[0] not in curv:
                            check_2 = 1
                            curv.append(line_2[0])
                if check_1 == 0:
                    if line_1 not in segments_cond_tangl:
                        segments_cond_tangl.append(line_1)
                if check_2 == 0:
                    if line_2 not in segments_cond_tangl:
                        segments_cond_tangl.append(line_2)
            else:
                segments_cond_tangl.append(line_1)
                segments_cond_tangl.append(line_2)
        j = 0
        while j < len(segments_cond_tangl):
            changes = True
            while changes:
                changes = False
                for segm_1 in segments_cond_tangl:
                    for segm_2 in [item for item in segments_cond_tangl if item != segm_1]:
                        if len(set(segm_1) & set(segm_2)) != 0:
                            changes = True
                            segm_1 += segm_2
                            segments_cond_tangl.remove(segm_2)
            j += 1
        for i in range(0, len(segments_cond_tangl)):
            segments_cond_tangl[i] = list(set(segments_cond_tangl[i]))
        for segm in segments_cond_tangl:
            if len(segm) == 0:
                segments_cond_tangl.remove(segm)
        ####удалим сразу все "простые" окружности
        #print('segments')
        #print(segments_cond_tangl)
        simple_circles = []
        for curve in segments_cond_tangl:
            if len(set(curve) & set(border_ribs)) == 0:
                #segments_cond_tangl.remove(curve)
                simple_circles.append(curve)
                gamma += 1

        segments_cond_tangl_1 = segments_cond_tangl.copy()
        segments_cond_tangl = [item for item in segments_cond_tangl_1 if item not in simple_circles]
        #print('segments')
        #print(segments_cond_tangl)
        ####
        for segm in segments_cond_tangl:
            edges_for_del = []
            for edge in segm:
                if edge not in border_ribs:
                    edges_for_del.append(edge)
            for edge_fd in edges_for_del:
                segm.remove(edge_fd)
        ###ищем [m, n], где m - число пересечений с меридианом, n- с параллелью
        ##ищем n:
        #n = 0
        #print(bind_ribs)
        #for d_2 in up_d:
            #n += find_d_sgn_n(d_2, segments_cond_tangl)
        #for m_2 in up_m:
            #n += find_m_sgn(m_2, segments_cond_tangl)
        #n = int(n / 2)
        ##ищем m:
        #m = 0
        #for d_2 in left_d:
            #m += find_d_sgn_m(d_2, segments_cond_tangl)
        #for m_2 in left_p:
            #m += find_p_sgn(m_2, segments_cond_tangl)
        #m = int(-m / 2)
        #print('[m, n]')
        #print([m, n])
        m = 0
        n = 0
        for segm in segments_cond_tangl:
            if len(segm) == 0:
                segments_cond_tangl.remove(segm)
        curves_cond = closing(segments_cond_tangl)
        #print('condition')
        #print(condition)
        #print('кривые')
        #print(curves_cond)
        for wind in curves_cond:
            sections = [wind[i: i + 2] for i in range(0, len(wind), 2)]
            #print('winding')
            #print(wind)
            #print(sections)
            m_1 = 0
            n_1 = 0
            for sec in sections:
                x_1 = coords(sec[0])
                x_2 = coords(sec[1])
                m_1 += x_2[0] - x_1[0]
                n_1 += x_2[1] - x_1[1]
                if sec[1] in up_d:
                    m_1 += 1 - x_2[0]
                    n_1 += 1 - x_2[0]
                if sec[1] in left_d:
                    m_1 += -(x_2[1])
                    n_1 += -(x_2[1])
                #print('section')
                #print(sec)
                #print([x_1, x_2])
            #print('[m_1, n_1]')
            #print([m_1, n_1])
            #print('[m, n]')
            #print([m_1, n_1])
            m_1 = round(m_1)
            n_1 = round(n_1)
            if n_1 < 0:
                m_1 = -m_1
                n_1 = -n_1
            elif n_1 == 0 and m_1 < 0:
                m_1 = -m_1
            #print('обрабатываемая кривая')
            #print(wind)
            m += int(m_1)
            n += int(n_1)
            #print('winding')
            #print(wind)
            #print('[m, n]')
            #print([m_1, n_1])
        #print('[m, n]')
        #print([m, n])
        #подсчет числа окружностей:
        gamma += len(curves_cond) - gcd(m, n)
        #print('gamma')
        #print(gamma)
        ###
        ###формирование многочлена
        f = ((-a)**(-3*sw)) * (a ** (alpha - beta)) * (-a ** 2 - a ** (-2)) ** gamma
        f = expand(f)
        #print('f')
        #print(f)
        if [m, n] in all_windings:
            id = all_windings.index([m, n])
            all_cond_poly[id] = expand(all_cond_poly[id] + f)
        else:
            all_windings.append([m, n])
            all_cond_poly.append(f)
        ###
    #print(all_windings)
    #print('sw')
    #print(sw)
    ind = index_lexmin_poly(all_cond_poly)
    #print(ind)
    #print(all_windings)
    all_windings = minimizing_vectors(all_windings)
    #print(all_windings)
    return all_cond_poly, all_windings


        ####
print('//////////////////////////////////////////////////////////////////////')
########КЛАССИФИКАЦИЯ ЗАЦЕПЛЕНИЙ
a = Symbol('a')
f_1 = a


def check_for_not_feature(mesh):
    #проверка на отсутствие "висячих" компонент зацепления и на отсуствие мест, где можно применить третье преобразование Рейдермейстера
    #аргументом является представление тангл+звезда
    def find_neigh(VE, vert, rib_1):
        for i in range(0, len(VE)):
            if rib_1 in VE[i] and i != (vert - 1):
                return i + 1
        if VE[vert - 1].count(rib_1) == 2:
            return vert
        return 0
    ve_tangl = mesh[0]
    ve_mesh = tangle_join_star(ve_tangl, mesh[1], mesh[2])
    ###третье преобразование Рейдермейстера
    for i in range(0, len(ve_tangl)):
        v_1 = i + 1
        incoming = ve_tangl[i]
        for j in range(0, 4):
            r = incoming[j]
            w_1 = find_neigh(ve_tangl, v_1, r)
            w_2 = find_neigh(ve_tangl, v_1, incoming[(j + 1) % 4])
            if w_1 == w_2 and w_1 != 0:
                incoming_1 = ve_tangl[w_1 - 1]
                q_1 = incoming.index(r)
                q_2 = incoming_1.index(r)
                if q_1 in [0, 2] and q_2 in [0, 2]:
                    return 0
                if q_1 in [1, 3] and q_2 in [1, 3]:
                    return 0
    ###
    ###наличие "висячих" компонент:
    cts = components(ve_mesh)
    for ct in cts:
        check_list = []
        for v in ve_mesh:
            ct_in_v = [vert for vert in v if vert in ct]
            if len(ct_in_v) != 0:
                for r in ct_in_v:
                    if v.index(r) in [0, 2]:
                        check_list.append(1)
                    else:
                        check_list.append(0)
        if len(set(check_list)) == 1:
            return 0
    ###
    return 1
def kauff_equal(n):
    def check_for_equal_kauff(kf_1, kf_2):
        #проверяет два многочлена на эквивалентность
        pol_1 = kf_1[0]
        pol_2 = kf_2[0]
        wind_1 = kf_1[1]
        wind_2 = kf_2[1]
        indexes = check_fe_canon_vect(wind_1, wind_2)
        if indexes == 0:
            return 0
        else:
            for i in range(0, len(indexes)):
                if simplify(expand(pol_1[i] - pol_2[indexes[i]])) != 0:
                    return 0
        return 1
    def check_for_equal_kauff_subs(kf_1, kf_2):
        #проверяет два многочлена на эквивалентность
        pol_1 = kf_1[0]
        pol_2 = kf_2[0]
        wind_1 = kf_1[1]
        wind_2 = kf_2[1]
        indexes = check_fe_canon_vect(wind_1, wind_2)
        if indexes == 0:
            return 0
        else:
            for i in range(0, len(indexes)):
                if simplify(expand(pol_1[i].subs(a, a**(-1)) - pol_2[indexes[i]])) != 0:
                    return 0
        return 1
    a = Symbol('a')
    meshings = generation_meshing(n)[0]

    ###
    ###возвращает:
    all_kauff_polynoms = [] #все различные многочлены
    equal_kauff_meshings = [] #все зацепления, соответствующие равным многочленам (на i- том месте лежит список всех зацеплений с многочленом all_kauff_polynoms[i])
    ###
    for mesh in meshings:
        mesh_kpoll = kauffman_polynomial(mesh)
        mesh_kpoll_rev = kauffman_polynomial([reverse(mesh[0]), mesh[1], mesh[2]])
        #проверяем, есть ли в all_kauff_polynoms многочлен этого зацепления mesh
        check = 0
        if len(all_kauff_polynoms) == 0:
            all_kauff_polynoms.append(mesh_kpoll)
            equal_kauff_meshings.append([mesh])
            check = 1
        else:
            for i in range(0, len(all_kauff_polynoms)):
                kpoll = all_kauff_polynoms[i]
                check_1 = check_for_equal_kauff(mesh_kpoll, kpoll)
                check_2 = check_for_equal_kauff_subs(mesh_kpoll, kpoll)
                check_3 = check_for_equal_kauff(mesh_kpoll_rev, kpoll)
                check_4 = check_for_equal_kauff_subs(mesh_kpoll_rev, kpoll)
                if (1 in [check_1, check_2, check_3, check_4]) and check == 0:
                    equal_kauff_meshings[i].append(mesh)
                    check = 1
        if check == 0:
            all_kauff_polynoms.append(mesh_kpoll)
            equal_kauff_meshings.append([mesh])
    return all_kauff_polynoms, equal_kauff_meshings
ok = 0
if ok == 1:
    kauff = kauff_equal(4)
    all_poly = kauff[0]
    all_eq_mesh = kauff[1]
    all_eq_mesh_ve = []
    j = 1
    #f = open('C:\\Users\\Lenovo\\Desktop\\Диплом\\class_alter_tangl\\meshings\\4diagrams_1.txt', 'a')
    for i in range(0, len(all_poly)):
        #print('полином')
        #print(all_poly[i])
        #print('все зацепления с этим полиномом')
        for mesh in all_eq_mesh[i]:
            #print('///')
            #print('номер')
            #print(j)
            #print(mesh)
            all_eq_mesh_ve.append(tangle_join_star(mesh[0], mesh[1], mesh[2]))
            if j == 40:
                print(mesh[0])
                print(reverse(mesh[0]))
                print(kauffman_polynomial(mesh))
                print(kauffman_polynomial([reverse(mesh[0]), mesh[1], mesh[2]]))
            j += 1

            #print('///')
    #print('ОБЩЕЕЕ ЧИСЛО МНОГОЧЛЕНОВ')
    #print(len(all_poly))
    #print('общее число зацеплений')
    #print(len(all_eq_mesh_ve))
numbers_for_del_5 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 25, 27, 28, 32, 35, 36, 39, 40, 41, 42, 49, 50, 53, 54, 59, 74, 77, 79, 82,
                     85, 87, 112, 113, 114, 115, 120, 121, 122, 141, 142, 143, 144, 159, 161, 164, 165, 166, 167, 170, 171, 172, 173, 174,
                     175, 176, 177, 181, 182, 189, 191, 193, 195, 196, 198, 199, 200, 213, 214, 215, 221, 222, 223, 224, 234,
                     235, 244, 245, 254, 255, 256, 257, 262, 264, 267, 268, 272, 273, 276, 279, 280, 281, 284, 286, 287, 289, 292, 293,
                     295, 296, 297, 298, 299, 302, 312, 313, 314, 315, 317, 319, 335, 345, 346, 347, 348, 351, 352, 354, 355, 359,
                     360, 361, 369, 370, 373, 374, 375, 376, 377, 378, 379, 383, 385, 386, 387, 391, 438, 439, 442, 443, 444, 445, 448,
                     450, 451, 454, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 508, 509, 510, 511, 517, 518, 519, 520, 521, 522, 523,
                     524, 526, 527, 528, 531, 532, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 550]
def kauff_five():
    def check_for_equal_kauff(kf_1, kf_2):
        #проверяет два многочлена на эквивалентность
        pol_1 = kf_1[0]
        pol_2 = kf_2[0]
        wind_1 = kf_1[1]
        wind_2 = kf_2[1]
        indexes = check_fe_canon_vect(wind_1, wind_2)
        if indexes == 0:
            return 0
        else:
            for i in range(0, len(indexes)):
                if simplify(expand(pol_1[i] - pol_2[indexes[i]])) != 0:
                    return 0
        return 1
    def check_for_equal_kauff_subs(kf_1, kf_2):
        #проверяет два многочлена на эквивалентность
        pol_1 = kf_1[0]
        pol_2 = kf_2[0]
        wind_1 = kf_1[1]
        wind_2 = kf_2[1]
        indexes = check_fe_canon_vect(wind_1, wind_2)
        if indexes == 0:
            return 0
        else:
            for i in range(0, len(indexes)):
                if simplify(expand(pol_1[i].subs(a, a**(-1)) - pol_2[indexes[i]])) != 0:
                    return 0
        return 1
    kauff = kauff_equal(5)
    all_poly = kauff[0]
    all_eq_mesh = kauff[1]
    all_eq_mesh_ve = []
    all_eq_mesh_tangl = []
    number = 1
    not_use_poly_ind = []
    one_uniq = []
    for i in range(0, len(all_poly)):
        eq_ve = []
        eq_tangl = []
        for j in range(0, len(all_eq_mesh[i])):
            if number not in numbers_for_del_5:
                eq_ve.append(tangle_join_star(all_eq_mesh[i][j][0], all_eq_mesh[i][j][1], all_eq_mesh[i][j][2]))
                eq_tangl.append(all_eq_mesh[i][j])
            number += 1
        if len(eq_ve) > 1:
            all_eq_mesh_ve.append(eq_ve.copy())
            all_eq_mesh_tangl.append(eq_tangl.copy())
        elif len(eq_ve) == 1:
            not_use_poly_ind.append(i)
            one_uniq += eq_ve
        else:
            not_use_poly_ind.append(i)

    all_poly_use = [all_poly[i] for i in range(0, len(all_poly)) if i not in not_use_poly_ind]
    kauff_2_poly, kauff_2_ve = kauff_equal(2)
    kauff_3_poly, kauff_3_ve = kauff_equal(3)
    kauff_4_poly, kauff_4_ve = kauff_equal(4)
    for i in range(0, len(all_poly_use)):
        poly = all_poly_use[i]
        for j in range(0, len(kauff_2_poly)):
            for j_1 in range(0, len(kauff_2_ve[j])):
                canon_mesh = kauff_2_ve[j][j_1]
                poly_1 = kauffman_polynomial(canon_mesh)
                poly_2 = kauffman_polynomial([reverse(canon_mesh[0]), canon_mesh[1], canon_mesh[2]])
                check_1 = check_for_equal_kauff(poly, poly_1)
                check_2 = check_for_equal_kauff(poly, poly_2)
                check_3 = check_for_equal_kauff_subs(poly, poly_1)
                check_4 = check_for_equal_kauff_subs(poly, poly_2)
                if 1 in [check_1, check_2, check_3, check_4]:
                    all_eq_mesh_ve[i] += [tangle_join_star(canon_mesh[0], canon_mesh[1], canon_mesh[2])]
                    all_eq_mesh_tangl[i] += [canon_mesh]
    for i in range(0, len(all_poly_use)):
        poly = all_poly_use[i]
        for j in range(0, len(kauff_3_poly)):
            for j_1 in range(0, len(kauff_3_ve[j])):
                canon_mesh = kauff_3_ve[j][j_1]
                poly_1 = kauffman_polynomial(canon_mesh)
                poly_2 = kauffman_polynomial([reverse(canon_mesh[0]), canon_mesh[1], canon_mesh[2]])
                check_1 = check_for_equal_kauff(poly, poly_1)
                check_2 = check_for_equal_kauff(poly, poly_2)
                check_3 = check_for_equal_kauff_subs(poly, poly_1)
                check_4 = check_for_equal_kauff_subs(poly, poly_2)
                if 1 in [check_1, check_2, check_3, check_4]:
                    all_eq_mesh_ve[i] += [tangle_join_star(canon_mesh[0], canon_mesh[1], canon_mesh[2])]
                    all_eq_mesh_tangl[i] += [canon_mesh]
    for i in range(0, len(all_poly_use)):
        poly = all_poly_use[i]
        for j in range(0, len(kauff_4_poly)):
            for j_1 in range(0, len(kauff_4_ve[j])):
                canon_mesh = kauff_4_ve[j][j_1]
                poly_1 = kauffman_polynomial(canon_mesh)
                poly_2 = kauffman_polynomial([reverse(canon_mesh[0]), canon_mesh[1], canon_mesh[2]])
                check_1 = check_for_equal_kauff(poly, poly_1)
                check_2 = check_for_equal_kauff(poly, poly_2)
                check_3 = check_for_equal_kauff_subs(poly, poly_1)
                check_4 = check_for_equal_kauff_subs(poly, poly_2)
                if 1 in [check_1, check_2, check_3, check_4]:
                    all_eq_mesh_ve[i] += [tangle_join_star(canon_mesh[0], canon_mesh[1], canon_mesh[2])]
                    all_eq_mesh_tangl[i] += [canon_mesh]
    numbers_for_del_uniq = [2, 3, 21, 48, 58, 59, 64, 69, 72, 83, 89, 90, 103, 122]
    numbers_not_uniq = [2, 4, 5, 7, 9, 11, 13, 15, 17, 19, 22, 23, 25, 28, 31, 34, 37, 39,
                        42, 45, 48, 52, 54, 59, 62, 64, 66, 67, 69, 71, 73, 75, 77, 79,
                        81, 83, 85, 89, 91, 93, 100, 102, 104, 106, 108, 110, 111, 116, 117,
                        120, 122, 124, 126, 127, 130, 132, 133, 135, 137, 139, 141, 142, 143,
                        145, 146, 148, 153, 155, 157, 159, 160, 162, 164, 169, 176, 194, 196, 198]
    #wtf(one_uniq, 'uniq_poly')
    not_uniq = []
    for i in range(0, len(all_eq_mesh_ve)):
        not_uniq += all_eq_mesh_ve[i]
    #wtf(not_uniq, 'not_uniq')
    #финал для записи
    one_uniq_final = [one_uniq[j] for j in range(0, len(one_uniq)) if (j + 1) not in numbers_for_del_uniq]
    not_uniq_final = [not_uniq[j] for j in range(0, len(not_uniq)) if (j + 1) in numbers_not_uniq]
    final = one_uniq_final + not_uniq_final
    wtf(final, 'final_five')
    #f = open('C:\\Users\\Lenovo\\Desktop\\Диплом\\Защита\\Результаты для защиты\\five_vert\\not_uniq_equal.txt', 'a')
    #j = 1
    #for i in range(0, len(all_poly_use)):
        #for mesh in all_eq_mesh_tangl[i]:
            #if j == 87:
                #print(87)
                #print(all_poly_use[i])
                #print(mesh)
            #if j == 88:
                #print(88)
                #print(all_poly_use[i])
                #print(mesh)

            #j += 1
    #print('4-вершинные многочлены')
    #j = 1
    #f = open('C:\\Users\\Lenovo\\Desktop\\Диплом\\Защита\\Результаты для защиты\\five_vert\\not_uniq_equal.txt', 'a')
    #for eqs in all_eq_mesh_ve:
        #f.write('[')
        #for eq in eqs:
            #f.write(str(j))
            #if len(eq) == 4:
                #f.write('(4)')
            #elif len(eq) == 3:
                #f.write('(3)')
            #elif len(eq) == 2:
                #f.write('(2)')
            #f.write(', ')
            #j += 1
        #f.write(']')
        #f.write('\n')
    #f.close()



    #print(all_poly_use)
    #print(all_eq_mesh_ve)
    return 0

kauff_five()


#print(kauffman_polynomial(mesh_1))
#print(kauffman_polynomial(mesh_2))
#all_eq = [all_eq_mesh_ve[i] for i in range(0, len(all_eq_mesh_ve)) if i != 2]
#wtf(all_eq_mesh_ve)
indexes_4 = [1, 11, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 30, 32, 34, 35, 37, 38, 39, 40, 44,
           46, 49, 52, 53, 57, 59, 62, 64, 66, 68, 70, 71, 72]

numbers_for_del_5 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 25, 27, 28, 32, 35, 36, 39, 40, 41, 42, 49, 50, 53, 54, 59, 74, 77, 79, 82,
                     85, 87, 112, 113, 114, 115, 120, 121, 122, 141, 142, 143, 144, 159, 161, 164, 165, 166, 167, 170, 171, 172, 173, 174,
                     175, 176, 177, 181, 182, 189, 191, 193, 195, 196, 198, 199, 200, 213, 214, 215, 221, 222, 223, 224, 234,
                     235, 244, 245, 254, 255, 256, 257, 262, 264, 267, 268, 272, 273, 276, 279, 280, 281, 284, 286, 287, 289, 292, 293,
                     295, 296, 297, 298, 299, 302, 312, 313, 314, 315, 317, 319, 335, 345, 346, 347, 348, 351, 352, 354, 355, 359,
                     360, 361, 369, 370, 373, 374, 375, 376, 377, 378, 379, 383, 385, 386, 387, 391, 438, 439, 442, 443, 444, 445, 448,
                     450, 451, 454, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 508, 509, 510, 511, 517, 518, 519, 520, 521, 522, 523,
                     524, 526, 527, 528, 531, 532, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 550]

#sort_eq_mesh_ve = []
#for i in [num - 1 for num in indexes_4]:
    #sort_eq_mesh_ve.append(all_eq_mesh_ve[i])
#print(len(sort_eq_mesh_ve))
#wtf(sort_eq_mesh_ve)
#print('////////////////////////////////////////////////')
#print(64)
#print(kauffman_polynomial(mesh_1))
#print(66)
#print(kauffman_polynomial(mesh_2))
#print(67)
#print(kauffman_polynomial(mesh_3))
#print('pol_1')
#print(mesh_2)
#print('pol_2')
#print(mesh_11)
#mesh_2 = [[[10, 9, 3, 4], [5, 7, 8, 10], [7, 6, 9, 8], [1, 2, 6, 5]], [1, 1, 0], 1]
#mesh_11 = [[[10, 3, 6, 7], [5, 8, 9, 10], [8, 1, 2, 9], [4, 5, 7, 6]], [1, 1, 0], 1]
#print(mesh_11)
#print(kauffman_polynomial(mesh_2))
#wtf(all_eq_mesh_ve)
#print(mesh_1)
#print(mesh_2)
#print("//////////////////////////////////////////////////////////////////////")
ok_1 = 0

if ok_1 == 1:
    kauff = kauff_equal(4)
    all_poly = kauff[0]
    all_eq_mesh = kauff[1]
    all_eq_mesh_ve = []
    j = 1
    eq_numbers = []
    #f = open('C:\\Users\\Lenovo\\Desktop\\Диплом\\class_alter_tangl\\meshings\\4diagrams_1.txt', 'a')
    for i in range(0, len(all_poly)):
        print('полином')
        print(all_poly[i])
        print('все зацепления с этим полиномом')
        eq = []
        for mesh in all_eq_mesh[i]:
            print('///')
            print('номер')
            print(j)
            print(mesh)
            all_eq_mesh_ve.append(tangle_join_star(mesh[0], mesh[1], mesh[2]))
            if len(all_eq_mesh[i]) > 1:
                eq.append(j)
            j += 1

        if len(eq) > 0:
            eq_numbers.append(eq.copy())
    print('eq_numbers')
    print(eq_numbers)


