import altair as alt
import numpy as np
import pandas as pd
from utils.constants import *
import copy


vars_lvl = {'level1':['FlightsClimbed','StepCount','Awake',
                'HeartRateReserve','RestingHeartRate','VO2Max','HeartRateVariabilitySDNN','CardiacEffort'],
            'level2':['FlightsClimbedPaceMax', 'StepCountPaceMax', 'Asleep','WalkingHeartRateAverage','HeartRate','BasalEnergyBurned','AppleStandTime','FlightsClimbedPaceMean']}

def make_timeline_plot2old(df,palette_long,timevar='days'):

    base = alt.Chart(df).encode(
        # alt.X("startTime:T", axis=alt.Axis(format="%Y-%B"))
        #     .title("Time"),
        alt.X(f"{timevar}:Q").title(f"{timevar} from consent"),
        alt.Y("patient:N",sort = 'ascending')
        # alt.Y("patient:N",sort = alt.SortField('order'))
            #   [{"field": "Group","order":"ascending"},
            #                     {"op": "min", "field": "startTime","order":"descending"}])
                                
            .axis(offset=0, ticks=False,labels=False, minExtent=0, domain=False)
            .title("Participant")
    )
# .transform_calculate(
#     # order='(datum.Group == "PAH" ? 100000 : 200000) * (5000+min(datum.startTime))'  
#     order=f'datum.Group_order * 100000 + datum.time_order'  
#         )
    line_first = base.mark_line().encode(
        detail="patient:N",
        opacity=alt.value(.8),
        color=alt.Color("devices:N").scale(range=['#2C75FF','black','red']),
    ).transform_filter(alt.FieldOneOfPredicate(field="measurement", oneOf=["first", "last",'withdrawn','dead']))

    point_base = base.mark_point(filled=True).encode(
        opacity=alt.value(1),
        shape = alt.Shape("measurement").scale(domain=["first", "last",'withdrawn','dead','diagnosis'],
                                               range=['triangle-left', 'triangle-right','triangle', 'cross','circle']).title('timepoint'),
        tooltip=['patient',f'{timevar}:Q','measurement',"Group"]).transform_filter(alt.FieldOneOfPredicate(field="measurement", 
                                                                                                        oneOf=["first", "last",'withdrawn','dead','diagnosis']))
    
    # caps = point_base.encode(color=alt.Color("Group",sort = alt.SortField('Group_order')).scale(domain=list(palette_long.keys()),range=list(palette_long.values())),
    #     size=alt.value(50))
    keys_present = [k for k in palette_with_us.keys() if k in list(df['Group'].unique())]
    palette_now = {k:v for k,v in palette_with_us.items() if k in keys_present}
    point = point_base.encode(color=alt.Color("Group",sort = alt.SortField('Group_order')).scale(domain=list(palette_now.keys()),range=list(palette_now.values())),
        size=alt.condition("datum['measurement'] == 'first' || datum['measurement'] == 'last'",
        alt.value(50),  # Color for 'apple'
        alt.value(100)   # Color for other names
    ))
    
    reference_line= alt.Chart(pd.DataFrame({'x':[0]})).mark_rule(color='black', size=1).encode(x='x')    
    # (point + point2).resolve_scale(color='independent')
    chart = (line_first + point +  reference_line).resolve_scale(color='independent').properties(
                width=800,
                height=1200
            ).configure_axis(labelFontSize=20,titleFontSize=22
            ).configure_title(fontSize=20, 
            anchor="middle").configure_legend(titleColor='black', 
            titleFontSize=20,labelFontSize=20).interactive()
    return chart

def make_plot_allvars(df,device,by='Group',opacity=.5,color_range = {'DC':"#FFC107","PH":"#1E88E5","Healthy":"#004D40"}):

    highlight = alt.selection_point(
            on="click", fields=["Group"], nearest=True
)
    line = alt.Chart(df).mark_line().encode(
        x='months:Q',
        y='value:Q',
        color=alt.Color('Group:N').scale(domain = list(color_range.keys()),range=list(color_range.values())),
        tooltip=['months', 'value', 'Group',by],
        detail = by,
        opacity=alt.value(opacity)
    ).add_params(
    highlight
)

    chart = alt.vconcat()
    for lvl,vars in vars_lvl.items():
        print(lvl,vars)
        row = alt.hconcat()
        for variable in vars:#df2.variable.unique():
            row |= line.transform_filter(
                alt.FieldEqualPredicate(field="variable", equal=variable)
            ).transform_filter(
                alt.FieldEqualPredicate(field="device", equal=device)
            ).properties(
            title=variable
        )
        chart &= row

    return chart.resolve_scale(y='independent').interactive()

def make_plot_allvars_ci(df,df2,device,color_range,xscale=None):
    # base = alt.Chart(df)
    varxscales = {0: ['AppleStandTime', 'Awake','BasalEnergyBurned', 'RestingHeartRate', 'WalkingHeartRateAverage', 'HeartRate', 'HeartRateReserve', 'VO2Max', 'HeartRateVariabilitySDNN','Asleep','FlightsClimbedPaceMax','StepCountPaceMax','FlightsClimbedPaceMean','CardiacEffort'], 
                  1: ['StepCount', 'FlightsClimbed']}
    varxscales = {variable: level for level, variables in varxscales.items() for variable in variables}

    line = alt.Chart(df2).mark_line().encode(
        y=alt.Y('value:Q'),
        color=alt.Color('Group:N').scale(domain = list(color_range.keys()),range=list(color_range.values())),
        tooltip=['months', 'value', 'Group'],
        detail = 'Group',
        opacity=alt.value(1)
    )

    band = alt.Chart(df).mark_errorband(extent='ci').encode(
            y=alt.Y('value:Q'),
            color=alt.Color('Group:N').scale(domain = list(color_range.keys()),range=list(color_range.values())),
            )
    reference_line = alt.Chart(pd.DataFrame({'x':[0]})).mark_rule(color='black', size=1).encode(x='x')

    chart = alt.vconcat()
    for lvl,vars in vars_lvl.items():
        row = alt.hconcat()
        for variable in vars:#df2.variable.unique():
            xscalenow = xscale[varxscales[variable]] if xscale else [0,df2.query(f"variable == '{variable}'").months.max()]
            row |= (band.encode(x=alt.X('months:Q', scale=alt.Scale(domain=xscalenow)))).transform_filter(
                alt.FieldEqualPredicate(field="variable", equal=variable)
            ).transform_filter(
                alt.FieldEqualPredicate(field="device", equal=device)
            ).properties(
            title=variable
        ) + (line.encode(x=alt.X('months:Q', scale=alt.Scale(domain=xscalenow)))).transform_filter(
                alt.FieldEqualPredicate(field="variable", equal=variable)
            ).transform_filter(
                alt.FieldEqualPredicate(field="device", equal=device)
            ) + reference_line
        chart &= row
    return chart.interactive()

def plot_distributions(df2,pvals_df,device,color_range):
    base = alt.Chart(df2)
    hist = base.mark_bar(opacity=0.5).encode(
            # x=alt.X("value:Q", bin=True),
             x=alt.X("value:Q", bin=alt.Bin(maxbins=50)),
            y=alt.Y('distinct(patient):Q',title="# of patients").stack(None),
            color=alt.Color('Group:N').scale(domain = list(color_range.keys()),range=list(color_range.values())),
            tooltip = ['Group',alt.Tooltip('value', bin=True),alt.Tooltip('distinct(patient)', title="patients")]
            )
    
    reference_line = base.mark_rule(opacity=1,size=2).encode(x="mean(value):Q",
                                                                  color=alt.Color('Group:N').scale(domain = list(color_range.keys()),range=list(color_range.values())),
                                                                  tooltip = ['Group',alt.Tooltip('mean(value):Q',format=',.1f').title('mean')])
    myplot = hist + reference_line

    chart = alt.vconcat()
    for lvl,vars in vars_lvl.items():
        row = alt.hconcat()
        for variable in vars:#df2.variable.unique():
            pvals = pvals_df.query(f'variable == "{variable}" and device == "{device}" and groups == "PH-Healthy"')
            if pvals.shape[0] == 0:
                pvals = {'pvalue':np.nan,'pvalue_adj_wh':np.nan}

            row |= myplot.transform_filter(
                alt.FieldEqualPredicate(field="variable", equal=variable)
            ).transform_filter(
                alt.FieldEqualPredicate(field="device", equal=device)
            ).properties(
            # title=f"{variable} - pval:10e{'%0.0f' % np.log10(pvals[variable])}"
            title=f"{variable} - pval:{'%0.2f' % pvals['pvalue']}({'%0.2f' % pvals['pvalue_adj_wh']} adj)"
        )
        chart &= row
    return chart.interactive()

# line = alt.Chart(df2).mark_line().encode(
#     x='months:Q',
#     y='mean(value):Q'
# )

# band = alt.Chart(df2).mark_errorband(extent='ci').encode(
#     x='months:Q',
#     y=alt.Y('value:Q'),
# )

# chart = (line+band).interactive()
# chart.save("imperial-410612/MHC/www/plot4.html")

    return chart

# specs_dict = json.loads(specs)
def height_fcn(var):
    if len(var) > 16:
        out = 23
        if len(var) > 20:
            out = 45
    else:
        out = 0
    return out

def allvar_plot(slopes,df22,df2now,units):
    
    heights = {'iPhone':[158,155,150],'Watch':[154,149,149]}
    df22['prediagnosis'] = df22['months_diagnosis'].apply(lambda x: 'Prediagnosis' if x<=0 else 'Postdiagnosis')
    # for vi,varsnow in enumerate([{k: rename_dict[k] for k in list(rename_dict)[:10]},{k: rename_dict[k] for k in list(rename_dict)[10:]}]):
    for vi,device in enumerate(['iPhone','Watch']):

        # varsnow = {k: rename_dict[k] for k in list(rename_dict) if rename_dict[k] in df22.query(f'device == "{device}"')['variable'].unique()}
        for letter, varsnow in allvarplot_dict[device].items():
            if letter == 'C':# ignore sleep
                continue

            # rows
            charts,height_addition = [],[]
            for vii,var in enumerate(varsnow.values()):

                height_addition.append(height_fcn(var))
                # color =  alt.Color('Group:N',scale=alt.Scale(domain=list(palette.keys()),range=list(palette.values()))) if vii != len(varsnow.values())-1 else alt.Color('Group:N',scale=alt.Scale(domain=list(palette.keys()),range=list(palette.values())),legend=alt.Legend(orient='bottom'))
                
                # plot the distributions
                xtitlenow = 'Value' if vii == len(varsnow.values())-1 else None
                titlenow = 'Distributions' if vii == 0 else ''
                # df22 = df22.query('Group != "PH" or months_diagnosis<1 or variable in ("Awake","Asleep") or (variable == "Active Energy" and device == "iPhone")')
                left_chart = alt.Chart(df22.query(f'device == "{device}" and variable == "{var}"')).mark_boxplot(opacity=1,extent='min-max').encode(
                    y=alt.Y('Group:N',sort=['DC','PH','Healthy'], axis=alt.Axis(labels=False)).title(var),
                    x=alt.X('value:Q', axis=alt.Axis(tickCount=3)).title(xtitlenow),#, scale=alt.Scale(domain=[0, 1])
                    color=alt.Color('Group:N',scale=alt.Scale(domain=list(palette.keys()),range=list(palette.values()))),
                    # detail='prediagnosis',
                    tooltip=['variable','Group','value_minmax'],
                    # row=alt.Row('variable:N', title=None,header=alt.Header(labelFontSize=20),sort=list(varsnow.values()))
                ).transform_filter(alt.FieldEqualPredicate(field='variable', 
                    equal=var)).transform_filter(alt.FieldEqualPredicate(field='device', 
                    equal=device)).properties(title = titlenow,width=500,height=heights[device][0])
                    # # oneOf=list(varsnow.values()))).properties(width=500,height=120)
                    
                
                xtitlenow = 'Months from diagnosis' if vii == len(varsnow.values())-1 else None
                titlenow = 'Mean values over time' if vii == 0 else ''
                # plot the trendlines
                df2nownow = df2now.query(f'device == "{device}" and variable == "{var}"')
                base2 = alt.Chart(df2nownow).transform_filter(alt.FieldRangePredicate(field='months', 
                                            range=[-12,12]))
                line2 = base2.mark_line(opacity=1).encode(y=alt.Y('mean:Q',axis=alt.Axis(tickCount=3)).scale(zero=False),
                                                          x=alt.X('months:Q',axis=alt.Axis(values=[-12,-6,0,6,12])).title(xtitlenow),
                    tooltip=['variable','Group','months','mean'],
                    color=alt.Color('Group:N',scale=alt.Scale(domain=list(palette.keys()),range=list(palette.values()))).title(None)
                    ).transform_filter(alt.FieldEqualPredicate(field='Group', equal='PH'))
                band2 = base2.mark_area(opacity=.3).encode(
                    x=alt.X('months:Q',axis=alt.Axis(values=[-12,-6,0,6,12])).title(xtitlenow),
                    y=alt.Y('upper:Q',axis=alt.Axis(tickCount=3)).scale(zero=False).title(units.query(f'variable == "{var}"')['unit_labels'].unique()[0]),
                    y2=alt.Y2('lower:Q').title(None),
                    color=alt.Color('Group:N',scale=alt.Scale(domain=list(palette.keys()),range=list(palette.values())))
                ).transform_filter(alt.FieldEqualPredicate(field='Group', equal='PH'))
                zero_line = base2.mark_rule(color='black', size=1).encode(x=alt.X('y'))
                mean_line = base2.mark_rule(size=3).encode(y=alt.Y('mean(mean):Q'),
                                                            color=alt.Color('Group:N',
                                                            scale=alt.Scale(domain=list(palette.keys()),
                                                                range=list(palette.values()))).title(None)).transform_filter(alt.FieldOneOfPredicate(field='Group', 
                                                                                                                                                        oneOf=['DC','Healthy']))

                middle_chart = (line2 + band2 + zero_line + mean_line).transform_filter(alt.FieldEqualPredicate(field='variable', 
                                                                                                equal=var)).transform_filter(alt.FieldEqualPredicate(field='device', 
                    equal=device)).properties(title = titlenow,width=500, height=heights[device][1])#.resolve_scale(y='independent')
                
                titlenow = r'% change from diagnosis' if vii == 0 else ''
                base = alt.Chart(slopes.query(f'device == "{device}" and variable == "{var}"')).transform_filter(alt.FieldEqualPredicate(field='Group', 
                                        equal='PH')).transform_filter(alt.FieldLTPredicate(field='slope', 
                                                        lt=.1)).transform_filter(alt.FieldGTPredicate(field='slope', 
                                                        gt=-.1)).transform_filter(alt.FieldOneOfPredicate(field='time_segment', 
                                                                                                                    oneOf=['-12to-9', '-9to-6','-6to-3', '-3to0', '0to3', '3to6','6to9','9to12']))#['-12to-6', '-6to0', '0to6', '6to12']

                dots = base.mark_circle(opacity=.9,size=45).encode(x=alt.X('time_segment:N', axis=alt.Axis(labelAngle=0),sort=['-12to-9', '-9to-6','-6to-3', '-3to0', '0to3', '3to6','6to9','9to12']).title(xtitlenow),#,axis=alt.Axis(tickCount=2)
                                                        y=alt.Y('slope:Q',axis=alt.Axis(tickCount=3,format='%')).title(None),
                                                        detail='patient',
                                                        tooltip=['variable','Group','time_segment','slope','patient'],
                    color=alt.Color('Group:N',scale=alt.Scale(domain=list(palette.keys()),range=list(palette.values()))))
                                                                
                avg_lines = base.mark_tick(opacity=1,size=40,thickness = 2.5).encode(x=alt.X('time_segment:N', axis=alt.Axis(labelAngle=0),sort=['-12to-9', '-9to-6','-6to-3', '-3to0', '0to3', '3to6','6to9','9to12']).title(xtitlenow),#,axis=alt.Axis(tickCount=2)
                                                        y=alt.Y('median(slope):Q',axis=alt.Axis(tickCount=3,format='%')).title(None),
                    color=alt.Color('Group:N',scale=alt.Scale(domain=list(palette.keys()),range=list(palette.values()))))
                reference_line = base.mark_rule(color='black', size=1).encode(y=alt.Y('y').title(None))
                upper_ref_line = base.mark_rule(color='black', size=1,strokeDash=[3,3]).encode(y=alt.Y('yu'))
                lower_ref_line = base.mark_rule(color='black', size=1,strokeDash=[3,3]).encode(y=alt.Y('yl'))
                right_chart = (dots + avg_lines + reference_line +upper_ref_line+lower_ref_line).properties(title = titlenow,width=500,height=heights[device][2]).transform_filter(alt.FieldEqualPredicate(field='variable', 
                                                                                                                    equal=var)).transform_filter(alt.FieldEqualPredicate(field='device', 
                    equal=device))
                # chart = (left_chart | middle_chart | right_chart) if vii == 0 else chart & (left_chart | middle_chart | right_chart)
                charts.append(copy.deepcopy([left_chart, middle_chart, right_chart]))

            charts = np.array(charts)
            for ii in range(3):
                for vii,chts in enumerate(charts):
                    height_addition_now = height_addition[vii] if ii != 0 else 0
                    chartnow = chts[ii] if vii == 0 else alt.vconcat(chartnow, chts[ii], spacing=height_addition_now)
                chartnow = chartnow.resolve_scale(x='independent') if ii == 0 else chartnow.resolve_scale(x='shared')
                chart = chartnow if ii == 0 else chart | chartnow
                    
                
            chart.configure_axis(labelFontSize=22,titleFontSize=20
                ).configure_title(fontSize=22, 
                anchor="middle").configure_legend(disable=True,titleColor='black', 
            titleFontSize=20,labelFontSize=20).save(f"frontend/www/plotxx{device}{letter}.html")
            print(f'{device}{letter} Done!')

def sleep_plot(df22,df2now,units):
    
    heights = {'iPhone':[158,155,150],'Watch':[154,149,149]}
    df22['prediagnosis'] = df22['months_diagnosis'].apply(lambda x: 'Prediagnosis' if x<=0 else 'Postdiagnosis')
    device  = 'Watch'
    letter = 'C'
    varsnow = allvarplot_dict[device][letter]

    # rows
    charts = []
    for vii,var in enumerate(varsnow.values()):
        
        # plot the distributions
        xtitlenow = 'Hrs' if vii == len(varsnow.values())-1 else None
        titlenow = 'Distributions' if vii == 0 else ''
        # df22 = df22.query('Group != "PH" or months_diagnosis<1 or variable in ("Awake","Asleep") or (variable == "Active Energy" and device == "iPhone")')
        left_chart = alt.Chart(df22.query(f'device == "{device}" and variable == "{var}"')).mark_boxplot(opacity=1,extent='min-max').encode(
            y=alt.Y('Group:N',sort=['DC','PH','Healthy'], axis=alt.Axis(labels=False)).title(var),
            x=alt.X('value:Q', axis=alt.Axis(tickCount=3)).title(xtitlenow),#, scale=alt.Scale(domain=[0, 1])
            color=alt.Color('Group:N',scale=alt.Scale(domain=list(palette.keys()),range=list(palette.values()))),
            # detail='prediagnosis',
            tooltip=['variable','Group','value'],
            # row=alt.Row('variable:N', title=None,header=alt.Header(labelFontSize=20),sort=list(varsnow.values()))
        ).transform_filter(alt.FieldEqualPredicate(field='variable', 
            equal=var)).transform_filter(alt.FieldEqualPredicate(field='device', 
            equal=device)).properties(title = titlenow,width=500,height=heights[device][0])
            # # oneOf=list(varsnow.values()))).properties(width=500,height=120)     
        
        xtitlenow = 'Months from diagnosis' if vii == len(varsnow.values())-1 else None
        titlenow = 'Mean values over time' if vii == 0 else ''
        # plot the trendlines
        base2 = alt.Chart(df2now.query(f'device == "{device}" and variable == "{var}"'))#.transform_filter(alt.FieldRangePredicate(field='months', range=[-12,12]))
        line2 = base2.mark_line(opacity=1).encode(y=alt.Y('mean:Q',axis=alt.Axis(tickCount=3)).scale(zero=False),
            x=alt.X('months:Q',axis=alt.Axis(tickCount=3)).title(xtitlenow),
            tooltip=['variable','Group','months','mean'],
            color=alt.Color('Group:N',scale=alt.Scale(domain=list(palette.keys()),range=list(palette.values()))).title(None)
            ).transform_filter(alt.FieldEqualPredicate(field='Group', equal='PH'))
        band2 = base2.mark_area(opacity=.3).encode(
            x=alt.X('months:Q',axis=alt.Axis(tickCount=3)).title(xtitlenow),
            y=alt.Y('upper:Q',axis=alt.Axis(tickCount=3)).scale(zero=False).title(units.query(f'variable == "{var}"')['unit_labels'].unique()[0]),
            y2=alt.Y2('lower:Q').title(None),
            color=alt.Color('Group:N',scale=alt.Scale(domain=list(palette.keys()),range=list(palette.values())))
        ).transform_filter(alt.FieldEqualPredicate(field='Group', equal='PH'))
        zero_line = base2.mark_rule(color='black', size=1).encode(x=alt.X('y'))
        mean_line = base2.mark_rule(size=3).encode(y=alt.Y('mean(mean):Q'),
                                                    color=alt.Color('Group:N',
                                                    scale=alt.Scale(domain=list(palette.keys()),
                                                        range=list(palette.values()))).title(None)).transform_filter(alt.FieldOneOfPredicate(field='Group', 
                                                                                                                                                oneOf=['DC','Healthy']))

        middle_chart = (line2 + band2 + zero_line + mean_line).transform_filter(alt.FieldEqualPredicate(field='variable', 
                                                                                        equal=var)).transform_filter(alt.FieldEqualPredicate(field='device', 
            equal=device)).properties(title = titlenow,width=500, height=heights[device][1])#.resolve_scale(y='independent')
        
            # chart = (left_chart | middle_chart | right_chart) if vii == 0 else chart & (left_chart | middle_chart | right_chart)
        charts.append(copy.deepcopy([left_chart, middle_chart]))

    charts = np.array(charts)
    for ii in range(2):
        for vii,chts in enumerate(charts):
            chartnow = chts[ii] if vii == 0 else chartnow & chts[ii]
        chartnow = chartnow.resolve_scale(x='independent')
        chart = chartnow if ii == 0 else chart | chartnow
            
        
    chart.configure_axis(labelFontSize=22,titleFontSize=20
        ).configure_title(fontSize=22, 
        anchor="middle").configure_legend(disable=True,titleColor='black', 
    titleFontSize=20,labelFontSize=20).save(f"frontend/www/plotxx{device}{letter}.html")

