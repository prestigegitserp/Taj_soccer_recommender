"""Plotly visualization primitives shared between notebook and dashboard."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from .spatial import pitch_access
from .schema import LENGTH,WIDTH

NAVY="#07111f"; GREEN="#1de3bb"; BLUE="#7198ff"; GOLD="#eabe63"; WHITE="#eaf3ff"

def pitch_shapes():
    shapes=[dict(type="rect",x0=-52.5,y0=-34,x1=52.5,y1=34,line_color="#78a59c",line_width=2),
            dict(type="line",x0=0,y0=-34,x1=0,y1=34,line_color="#78a59c",line_width=1),
            dict(type="circle",x0=-9.15,y0=-9.15,x1=9.15,y1=9.15,line_color="#78a59c",line_width=1)]
    for s in (-1,1):
        x0=-52.5 if s<0 else 36.0; x1=-36 if s<0 else 52.5
        shapes.append(dict(type="rect",x0=x0,y0=-20.16,x1=x1,y1=20.16,
                           line_color="#78a59c",line_width=1))
        xx0=-52.5 if s<0 else 47.; xx1=-47 if s<0 else 52.5
        shapes.append(dict(type="rect",x0=xx0,y0=-9.16,x1=xx1,y1=9.16,
                           line_color="#78a59c",line_width=1))
    return shapes

def pitch_figure(frame:pd.DataFrame,ball=None,heatmap=True,step=3.,title=None):
    fig=go.Figure()
    if heatmap and not frame.empty:
        access,away,X,Y=pitch_access(frame,step=step)
        fig.add_trace(go.Heatmap(x=X[0],y=Y[:,0],z=access,
            colorscale=[[0,"#405ab6"],[.5,"#152e38"],[1,"#15cba0"]],
            zmin=0,zmax=1,opacity=.68,
            colorbar=dict(title="Home access",tickfont=dict(color=WHITE),titlefont=dict(color=WHITE),
                          thickness=12,len=.5),hovertemplate="x=%{x:.1f}m, y=%{y:.1f}m<br>Home access index=%{z:.2f}<extra></extra>"))
    for team,color in (("Home",GREEN),("Away",BLUE)):
        sub=frame[frame.team==team]
        fig.add_trace(go.Scatter(x=sub.x,y=sub.y,mode="markers+text",
            text=sub.player_id.astype(str).str.slice(-3),textposition="middle center",
            textfont=dict(size=9,color=NAVY if team=="Home" else "#081320"),
            marker=dict(size=22,color=color,line=dict(color="#eef9ff",width=1)),
            name=team,customdata=sub[["player_id","is_detected"]].astype(str).to_numpy(),
            hovertemplate="Player %{customdata[0]}<br>observed=%{customdata[1]}<br>x=%{x:.1f},y=%{y:.1f}<extra>"+team+"</extra>"))
    if ball is not None and len(ball):
        br=ball.iloc[0]
        fig.add_trace(go.Scatter(x=[br.x],y=[br.y],mode="markers",
                    marker=dict(size=12,color=GOLD,line=dict(color="white",width=2)),
                    name="Ball",hovertemplate="Ball (%{x:.1f}, %{y:.1f})<extra></extra>"))
    fig.update_layout(title=title or "Tracking & spatial access",paper_bgcolor=NAVY,plot_bgcolor="#09291f",
        font=dict(color=WHITE,family="Arial"),height=585,
        margin=dict(l=20,r=20,t=58,b=20),legend=dict(orientation="h",y=1.08,x=.02),
        shapes=pitch_shapes(),xaxis=dict(range=[-56,56],visible=False,scaleanchor="y",scaleratio=1),
        yaxis=dict(range=[-37,37],visible=False),hovermode="closest")
    return fig
