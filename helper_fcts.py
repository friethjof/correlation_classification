

def get_interaction_str(df_par, index):
    """
    Create interaction string from run defined by index ind

    Args:
        df_par (DataFrame): inpur data frame with interaction parameters
        index (int): index of run
    
    Returns:
        str with interaction values
    """


    title_str = ''
    title_str += r'$g_{BB}=' + str(df_par.loc[index, 'gBB']) + '$, \\'
    title_str += r'$g_{CC}=' + str(df_par.loc[index, 'gCC']) + '$, \n'
    title_str += r'$g_{AB}=' + str(df_par.loc[index, 'gAB']) + '$, \n'
    title_str += r'$g_{AC}=' + str(df_par.loc[index, 'gAC']) + '$, \n'
    title_str += r'$g_{BC}=' + str(df_par.loc[index, 'gBC']) + '$ \n'
    
    return title_str

