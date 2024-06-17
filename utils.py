import ctypes

def flatten(t):
    return [item for sublist in t for item in sublist]

def chop(value, tol):
    return round(value/tol)*tol

def dictprint(dictionary, tabstring = ''):
    for key,value in dictionary.items():
        if type(value) is dict:
            tabstring = '   '
            print('\n')
            print(key)
            dictprint(value, tabstring)
            continue # interrupts for loop here and continues with next iteration
        print(tabstring, key, ':', value)  
    return

def dict_to_file(dictionary, filename, tabstring = '', w = True):
    output_string = str()
    for key,value in dictionary.items():
        if type(value) is dict:
            tabstring = '   '
            output_string += key+'\n'
            output_string += dict_to_file(value, filename, tabstring = tabstring, w = False)
            output_string += '\n'
            continue # interrupts for loop here and continues with next iteration
        output_string += f'{tabstring}{key}  {value}\n'

    if w == True:
        with open(filename, 'w') as f:
            f.write(output_string)
        return

    return output_string

def msgbox(text, title = 'Title', style = 0):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def IPOPT_global_options():

    options = {
        'ipopt': {
            'max_iter': 3000,
            # 'nlp_scaling_method': 'none',
            'linear_solver': 'mumps', # 'ma57',
            'ma57_pre_alloc': 10,
            'linear_system_scaling': 'none',
            'tol': 1e-6,
            'accept_every_trial_step': 'no',
            # 'mumps_permuting_scaling': 2,
            # 'mumps_pivot_order': 3,
            # 'mumps_scaling': 10,
            'fast_step_computation': 'no',
            # 'ma97scaling': 0,
            # 'line_search_method': 'cg-penalty',
            'print_level': 5,
            # 'watchdog_shortened_iter_trigger': 0,
            # 'warm_start_init_point': 'yes',
            # 'mu_init': 1e-3,
            # 'mu_oracle': 'probing',
            'alpha_for_y': 'primal',
            'mu_strategy': 'adaptive', # 'adaptive'; % 'monotone'; %
            'adaptive_mu_globalization': 'never-monotone-mode',
            'min_refinement_steps': 20,
            'max_refinement_steps' : 30
            },
        'print_time': 0,
        'error_on_fail': False
        }
    return options


def main():
    tabstring = ''
    # tabstring +='\t'
    tabstring +='hello'
    print('hello')
    print(tabstring)
    msgbox('This is a message box!', 'Title', 0)

if __name__ == "__main__":
    main()

