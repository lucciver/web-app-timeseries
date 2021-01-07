import numpy as np
import streamlit


def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def altair_chart(err_name, st, data):
    st.subheader('Comparision of infection growth')
    total_cases_graph  = alt.Chart(data).transform_filter(alt.datum.total_cases > 0
    ).mark_line().encode(x=alt.X('date', type='nominal', title = err_name),
   y=alt.Y('sum(total_cases):Q',  title='Confirmed cases'),
   color='Country',
   tooltip = 'sum(total_cases)',
   ).properties(
    width=1500,
    height=600
    ).configure_axis(
    labelFontSize=17,
    titleFontSize=20
    )
    st.altair_chart(total_cases_graph)
