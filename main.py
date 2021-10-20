import math
import matplotlib.pyplot as plt


class GraphEditor:
    def __init__(self):
        figure, plots = plt.subplots(nrows=3, ncols=1)
        self.gte_graph = plots[1]
        self.lte_graph = plots[2]
        self.solution_graph = plots[0]
        plt.figure(dpi=1200)

    def generate_gte_graph(self):
        self.gte_graph.set_xlabel('x')
        self.gte_graph.set_ylabel('GTE')

    def generate_lte_graph(self):
        self.lte_graph.set_xlabel('x')
        self.lte_graph.set_ylabel('LTE')

    def generate_solution_graph(self):
        self.solution_graph.set_xlabel('x')
        self.solution_graph.set_ylabel('y')

    @staticmethod
    def generate_legend():
        leg = plt.legend(["exact_solution", "euler", "improved_euler", "runge_kutta"],
                         loc='upper center', bbox_to_anchor=(0.5, 3.8), fancybox=True, ncol=4)
        colors = ["black", "green", "blue", "orange"]
        for i, j in enumerate(leg.legendHandles):
            j.set_color(colors[i])
        for handle in leg.legendHandles:
            handle.set_linewidth(5)

    def generate_figure(self):
        self.generate_gte_graph()
        self.generate_lte_graph()
        self.generate_solution_graph()
        self.generate_legend()

    @staticmethod
    def show_figure():
        plt.show()

    @staticmethod
    def save_graph():
        plt.savefig('filename.png')


class Equation:
    def __init__(self, x0, x, y0, graph_editor: GraphEditor):
        self.x0 = x0
        self.x = x
        self.y0 = y0
        self.graph_editor = graph_editor


class DifferentialEquation(Equation):
    def __init__(self, x0, x, y0, graph_editor: GraphEditor):
        Equation.__init__(self, x0, x, y0, graph_editor)

    @staticmethod
    def get_y_prime(x, y):
        return y / x + x * math.cos(x)

    @staticmethod
    def get_y(x):
        return x * math.sin(x) + 1

    @staticmethod
    def print_format(x, y):
        print('x:', '{:.2f}'.format(x), end=' ')
        print('y(', 'exact', '):', sep='', end=' ')
        print('{:.4f}'.format(y))

    def exact_solution(self, step):
        print('Exact solution:')
        k = self.x0

        while k < self.x + step:
            self.graph_editor.solution_graph.scatter(k, self.get_y(k), s=1, color='black', label='exact')
            self.print_format(k, self.get_y(k))
            k += step


class NumericalMethod:
    def __init__(self, step, differential_equation: DifferentialEquation, graph_editor: GraphEditor,
                 method_name, method_color):
        self.method_name = method_name
        self.method_color = method_color
        self.step = step
        self.differential_equation = differential_equation
        self.graph_editor = graph_editor

    @staticmethod
    def get_gte(y_current, y_exact):
        return abs(y_exact - y_current)

    @staticmethod
    def get_lte(y_from_exact, y_exact):
        return abs(y_exact - y_from_exact)

    def print_format(self, x, y, lte, gte):
        print('x:', '{:.2f}'.format(x), end=' ')
        print('y(', self.method_name, '):', sep='', end=' ')
        print('{:.4f}'.format(y), 'LTE:', '{:.4f}'.format(lte), 'GTE:', '{:.4f}'.format(gte))

    def supplement_graph(self, x, y, lte, gte):
        self.graph_editor.gte_graph.scatter(x, gte, s=1, color=self.method_color, label=self.method_name)
        self.graph_editor.lte_graph.scatter(x, lte, s=1, color=self.method_color, label=self.method_name)
        self.graph_editor.solution_graph.scatter(x, y, s=1, color=self.method_color, label=self.method_name,)


class EulerMethod(NumericalMethod):
    def __init__(self, step, differential_equation, graph_editor):
        NumericalMethod.__init__(self, step, differential_equation, graph_editor, 'euler', 'green')

    def euler_next(self, x_prev, y_prev):
        return y_prev + self.step * DifferentialEquation.get_y_prime(x_prev, y_prev)

    def method_implementation(self):
        print('Euler method:')
        y_prev = self.differential_equation.y0
        x_prev = self.differential_equation.x0
        k = self.differential_equation.x0 + self.step

        gte = self.get_gte(y_prev, DifferentialEquation.get_y(x_prev))
        lte = self.get_lte(y_prev, DifferentialEquation.get_y(x_prev))

        self.supplement_graph(x_prev, y_prev, lte, gte)
        self.print_format(x_prev, y_prev, lte, gte)

        while k <= self.differential_equation.x + self.step:
            y_next = self.euler_next(x_prev, y_prev)
            x_next = k

            gte = self.get_gte(y_next, DifferentialEquation.get_y(x_next))
            lte = self.get_lte(self.euler_next(x_prev, DifferentialEquation.get_y(x_prev)),
                               DifferentialEquation.get_y(x_next))

            self.supplement_graph(x_next, y_next, lte, gte)
            self.print_format(x_next, y_next, lte, gte)

            x_prev = x_next
            y_prev = y_next
            k += self.step


class ImproverEuler(NumericalMethod):
    def __init__(self, step, differential_equation, graph_editor):
        NumericalMethod.__init__(self, step, differential_equation, graph_editor, 'improved_euler', 'blue')

    def improved_euler_next(self, x_prev, y_prev):
        return y_prev + self.step * \
               DifferentialEquation.get_y_prime(x_prev + self.step / 2,
                                                y_prev + self.step / 2 * DifferentialEquation.get_y_prime(x_prev,
                                                                                                          y_prev))

    def method_implementation(self):
        print('Improved method:')
        y_prev = self.differential_equation.y0
        x_prev = self.differential_equation.x0
        k = self.differential_equation.x0 + self.step

        gte = self.get_gte(y_prev, DifferentialEquation.get_y(x_prev))
        lte = self.get_lte(y_prev, DifferentialEquation.get_y(x_prev))

        self.supplement_graph(x_prev, y_prev, lte, gte)
        self.print_format(x_prev, y_prev, lte, gte)

        while k <= self.differential_equation.x + self.step:
            y_next = self.improved_euler_next(x_prev, y_prev)
            x_next = k

            gte = self.get_gte(y_next, DifferentialEquation.get_y(x_next))
            lte = self.get_lte(self.improved_euler_next(x_prev, DifferentialEquation.get_y(x_prev)),
                               DifferentialEquation.get_y(x_next))

            self.supplement_graph(x_next, y_next, lte, gte)
            self.print_format(x_next, y_next, lte, gte)

            x_prev = x_next
            y_prev = y_next
            k += self.step


class RungeKutta(NumericalMethod):
    def __init__(self, step, differential_equation, graph_editor):
        NumericalMethod.__init__(self, step, differential_equation, graph_editor, 'runge_kutta', 'orange')

    def runge_kutta_next(self, x_prev, y_prev):
        k1 = DifferentialEquation.get_y_prime(x_prev, y_prev)
        k2 = DifferentialEquation.get_y_prime(x_prev + self.step / 2, y_prev + self.step * k1 / 2)
        k3 = DifferentialEquation.get_y_prime(x_prev + self.step / 2, y_prev + self.step * k2 / 2)
        k4 = DifferentialEquation.get_y_prime(x_prev + self.step, y_prev + self.step * k3)
        k_sum = k1 + 2 * k2 + 2 * k3 + k4

        return y_prev + self.step / 6 * k_sum

    def method_implementation(self):
        print('Runge Kutta method:')
        y_prev = self.differential_equation.y0
        x_prev = self.differential_equation.x0
        k = self.differential_equation.x0 + self.step

        gte = self.get_gte(y_prev, DifferentialEquation.get_y(x_prev))
        lte = self.get_lte(y_prev, DifferentialEquation.get_y(x_prev))

        self.supplement_graph(x_prev, y_prev, lte, gte)
        self.print_format(x_prev, y_prev, lte, gte)

        while k <= self.differential_equation.x + self.step:
            y_next = self.runge_kutta_next(x_prev, y_prev)
            x_next = k

            gte = self.get_gte(y_next, DifferentialEquation.get_y(x_next))
            lte = self.get_lte(self.runge_kutta_next(x_prev, DifferentialEquation.get_y(x_prev)),
                               DifferentialEquation.get_y(x_next))

            self.supplement_graph(x_next, y_next, lte, gte)
            self.print_format(x_next, y_next, lte, gte)

            x_prev = x_next
            y_prev = y_next
            k += self.step


X0 = math.pi
Y0 = 1
X = 4 * math.pi
STEP = 0.5

print('X0 = π, Y0 = 1, X = 4π, STEP = 0.5. Do you want to change this values? 1-yes, 2-no')
ans = input()
while True:
    if ans == '1':
        print('enter X0 value, please:')
        X0 = int(input())
        print('enter X0 value, please:')
        Y0 = int(input())
        print('enter X0 value, please:')
        X = int(input())
        print('enter STEP value, please:')
        STEP = int(input())
        print('thank you!')
        break
    elif ans == '2':
        print('ok')
        break
    else:
        print('incorrect answer: try again')
        ans = input()

gr_edit = GraphEditor()

diff_eq = DifferentialEquation(X0, X, Y0, gr_edit)
diff_eq.exact_solution(STEP)

euler_method = EulerMethod(STEP, diff_eq, gr_edit)
euler_method.method_implementation()

improved_euler = ImproverEuler(STEP, diff_eq, gr_edit)
improved_euler.method_implementation()

runge_kutta = RungeKutta(STEP, diff_eq, gr_edit)
runge_kutta.method_implementation()

gr_edit.generate_figure()
gr_edit.show_figure()
gr_edit.save_graph()
