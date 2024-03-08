class Calculator(object):
    def read(self) :
        '''read input from stdin'''
        return input('> ')
    
    def eval(self, string) :
        '''evaluates an infix arithmetic expression '''
        '''eliminam spatiile din sir'''
        string = ''.join(string.split())

        '''inainte sa citim sirul de caractere numarul nostru este 0
        iar operatorul implicit este "+"'''
        num = 0
        op = "+"

        '''cream o stiva in care sa tinem numerele folosite in calcularea expresiei'''
        stack = []

        '''cream o functie care sa ne ajute sa calculam expresia in functie de operatorul intalnit si valoarea curenta a variabile num
        functia adauga in stiva numere ce pot fi adunate pentru a ajunge la rezultatul final
        in cazul in care utilizatorul incearca sa imparta la 0 se arunca exceptia CalculatorException cu un mesaj specific'''
        def helper(op, num):
            if op == "+":
                stack.append(num)
            elif op == "-":
                stack.append(-num)
            elif op == "*":
                stack.append(stack.pop() * num)
            elif op == "/":
                if num == 0:
                    raise CalculatorException("Nu putem imparti la 0.")
                stack.append(int(stack.pop() / num))
        
        '''cream o bucla ce parcurge intreaga expresie'''
        # cazul 1 - ne aflam pe o cifra
        '''in aceasta situatie este posibil sa avem un numar format din doua sau mai multe cifre, motiv pentru care vom inmulti numarul descoperit pana in acel moment cu 10 si vom adauga cifra pe care ne aflam, astfel incat sa obtinem numarul corect
        spre exemplu, daca pana acum am descoperit cifra 3 (num = 3) si ne aflam pe cifra 4, inmultim 3 cu 10 si adunam cifra 4, obtinand numarul 34 (num = 34)'''
        # cazul 2 - ne aflam pe o paranteza deschisa
        '''adaugam in stiva operatorul descoperit anterior si resetam num si op la valorile implicite'''
        # cazul 3.1 - ne aflam pe un operator
        '''adaugam in stiva numarul cu ajutorul functiei helper, resetam num la valoarea implicita si op devine operatorul descoperit'''
        # cazul 3.2. - ne aflam pe o paranteza inchisa
        ''' adaugam in stiva numarul curent cu ajutorul functiei helper si parcurgem stiva pana dam de un operator (atunci am terminat de parcurs paranteza care se inchide la pozitia pe care ne aflam in sir)
        atat timp cat in stiva avem numere, le adaugam la num si le eliminam din aceasta (functia helper a adaugat in stiva numere astfel incat acestea sa poata fi adunate la final)
        cand am ajuns pe un operator schimbam valoarea lui op cu acesta si il eliminam din stiva
        in final apelam functia helper pentru a adauga in stiva rezultatul dintre paranteza si operandul dinaintea ei
        cand iesim din bucla resetam num la valoarea implicita si op devine ")" - aceasta valoare nu va influenta cu nimic executia functiei helper, intrucat nu am definit un caz pentru aceasta valoare'''
        # cazul 4 - ne aflam pe un caracter invalid
        '''daca utilizatorul a introdus in expresie un caracter invalid programul va arunca o exceptie de tipul CalculatorException cu un mesaj specific'''
        for i in range(len(string)):
            if string[i].isdigit():
                num = num * 10 + int(string[i])
            elif string[i] == "(":
                stack.append(op)
                num = 0
                op = "+"
            elif string[i] in ["+", "-", "*", "/", ")"]:
                helper(op, num)
                if string[i] == ")":
                    num = 0
                    while isinstance(stack[-1], int):
                        num += stack.pop()
                    op = stack.pop()
                    helper(op, num)
                num = 0
                op = string[i]
            else:
                raise CalculatorException("Expresia contine caractere invalide. Poti folosi doar cifre, paranteze sau operatori ('+', '-', '*', '/').")
        
        '''in final se proceseaza ultimul num cu ajutorul functiei helper si se insumeaza toate numerele din stiva'''
        helper(op, num)
        '''verificam daca a ramas vreun semn in stiva (in mod normal aceasta ar trebui sa contina doar numere)'''
        for element in stack:
            if isinstance(element, str):
                raise CalculatorException("Expresia are o forma invalida. Asigura-te ca ai formulat o expresie valida.")
        return sum(stack)

    def loop(self) :
        """read a line of input, evaluate and print it
        repeat the above until the user types 'quit'."""
        '''sirul citit este convertit in litere mici pentru a functiona indiferent de modul in care utilizatorul tasteaza cuvantul "quit"'''
        while True:
            line = self.read().lower()
            if line == 'quit':
                break
            result = self.eval(line)
            print(result)
    
class CalculatorException(Exception):
    pass

if __name__ == '__main__':
    calc = Calculator()
    calc.loop()