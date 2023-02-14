
class HornRule():
    def __init__(self,path,head,mode="ALL_VAR"):
        """
        Rule in horn-clause format, genaralized from concrete path
        :param path: concrete reasoning path, list of (relation,entity)
        :param head: horn-clause head, (e_s,r,e_t) 
        :param mode: rule generalization mode
            ALL_VAR: all entities in path and head are variables
            BODY_VAR:  only entities in body are variables
            CONSTANT_HEAD: entities are variables except the first one
        """
        self.body_length = len(path)-1
        self.body = []
        for r,e in path:
            if r == 2:
                self.body_length -= 1
                continue
            self.body.append([r,-1])
        if mode == "ALL_VAR":
            self.head = (-1,head[1],-1)
        elif mode == "CONSTANT_HEAD":
            self.head = (head[0],head[1],-1)
            self.body[0][1] = head[0]
        elif mode == "BODY_VAR":
            self.head = head
            self.body[0][1] = head[0]
            self.body[-1][1] = head[2]
        self.mode = mode
    
    def get_predicate(self,step):
        return (self.body[step][1],self.body[step+1][0],self.body[step+1][1])

    def get_rel_path(self):
        return [pair[0] for pair in self.body[1:]]

    def compatible(self,predicate,assignment):
        is_e1 = predicate[0] == -1 or predicate[0] == assignment[0]
        is_r = predicate[1] == -1 or predicate[1] == assignment[1]
        is_e2 = predicate[2] == -1 or predicate[2] == assignment[2]
        return is_e1 and is_r and is_e2
    
    def get_str_representation(self):
        body = ",".join(["{},{}".format(r,e) for [r,e] in self.body])
        head = "{},{},{}".format(self.head[0],self.head[1],self.head[2])
        return "->".join([body,head])