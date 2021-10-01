fig ,((ax1 ,ax2 ,ax3 ,ax4 ,ax5)) = plt.subplots( nrows=1 ,ncols=5 ,figsize=(21 ,3) )

width = 0.1
x = [5, 8, 10]
x_loc = [5, 6.5, 9]

# wired
RobustMPC = [3.664]
FastMPC = [3.58]
BBA=[3.57]
ADR = [3.68]
UDR_1 = [2.367]
UDR_2 = [3.28]
UDR_3 = [2.824]


RobustMPC_err = [0.11]
FastMPC_err = [0.13]
BBA_err=[0.17]
ADR_err = [0.14]
UDR_1_err = [0.17]
UDR_2_err = [0.19]
UDR_3_err = [0.28]

ax1.bar( x_loc[2] ,ADR ,yerr=ADR_err ,width=1 ,color='C2' ,label="GENET" )
ax1.bar( x_loc[1] ,RobustMPC ,yerr=RobustMPC_err ,width=1 ,color='C0' ,hatch='xx' ,alpha=1 ,label="RobustMPC" )
#ax1.bar( 5 ,FastMPC ,yerr=FastMPC_err ,width=1 ,color='C0' ,hatch='..' ,alpha=1 ,label="MPC" )
ax1.bar( x_loc[0] ,BBA ,yerr=BBA_err ,width=1 ,color='C0' ,hatch='--' ,alpha=1 ,label="BBA" )
# ax1.bar( 9 ,UDR_1 ,yerr=UDR_1_err ,width=1 ,color='grey' ,hatch='xx' ,alpha=0.7 ,label="UDR-1" )
# ax1.bar( 10 ,UDR_2 ,yerr=UDR_2_err ,width=1 ,color='grey' ,hatch='..' ,alpha=0.7 ,label="UDR-2" )
# ax1.bar( 11 ,UDR_3 ,yerr=UDR_3_err ,width=1 ,color='grey' ,hatch='--' ,alpha=0.7 ,label="UDR-3" )

ax1.set_xticks( []  )
ax1.spines['right'].set_visible( False )
ax1.spines['top'].set_visible( False )

ax1.set_title( "Wired" )
ax1.set_ylabel( 'Test reward' )

# Wifi
RobustMPC = [2.72]
FastMPC = [2.51]
BBA = [2.57]
ADR = [2.96]
UDR_1 = [1.35]
UDR_2 = [1.98]
UDR_3 = [1.64]

RobustMPC_err = [0.14]
FastMPC_err = [0.15]
BBA_err = [0.16]
ADR_err = [0.17]
UDR_1_err = [0.18]
UDR_2_err = [0.31]
UDR_3_err = [0.27]

ax2.bar( x_loc[2] ,ADR ,yerr=ADR_err ,width=1 ,color='C2' ,label="GENET" )
ax2.bar( x_loc[1] ,RobustMPC ,yerr=RobustMPC_err ,width=1 ,color='C0' ,hatch='xx' ,alpha=1 ,label="RobustMPC" )
# ax1.bar( 5 ,FastMPC ,yerr=FastMPC_err ,width=1 ,color='C0' ,hatch='..' ,alpha=1 ,label="MPC" )
ax2.bar( x_loc[0] ,BBA ,yerr=BBA_err ,width=1 ,color='C0' ,hatch='--' ,alpha=1 ,label="BBA" )
# ax2.bar( 9 ,UDR_1 ,yerr=UDR_1_err ,width=1 ,color='grey' , hatch='xx', alpha=0.7 , label="UDR-1" )
# ax2.bar( 10 ,UDR_2 ,yerr=UDR_2_err ,width=1 ,color='grey' , hatch='..', alpha=0.7,label="UDR-2" )
# ax2.bar( 11 ,UDR_3 ,yerr=UDR_3_err ,width=1 ,color='grey' ,hatch='--', alpha=0.7,label="UDR-3" )

ax2.set_xticks( []  )
ax2.spines['right'].set_visible( False )
ax2.spines['top'].set_visible( False )

ax2.set_title("WiFi")

# # 5G
RobustMPC = [1.59]
FastMPC = [1.27]
BBA = [1.37]
ADR = [1.85]
UDR_1 = [0.75]
UDR_2 = [1.04]
UDR_3 = [0.824]

RobustMPC_err = [0.22]
FastMPC_err = [0.21]
BBA_err = [0.13]
ADR_err = [0.18]
UDR_1_err = [0.18]
UDR_2_err = [0.11]
UDR_3_err = [0.17]

ax3.bar( x_loc[2] ,ADR ,yerr=ADR_err ,width=1 ,color='C2' ,label="GENET" )
ax3.bar( x_loc[1] ,RobustMPC ,yerr=RobustMPC_err ,width=1 ,color='C0' ,hatch='xx' ,alpha=1 ,label="RobustMPC" )
# ax1.bar( 5 ,FastMPC ,yerr=FastMPC_err ,width=1 ,color='C0' ,hatch='..' ,alpha=1 ,label="MPC" )
ax3.bar( x_loc[0] ,BBA ,yerr=BBA_err ,width=1 ,color='C0' ,hatch='--' ,alpha=1 ,label="BBA" )
# ax3.bar( 9 ,UDR_1 ,yerr=UDR_1_err ,width=1 ,color='grey' ,hatch='xx' ,alpha=0.7 ,label="UDR-1" )
# ax3.bar( 10 ,UDR_2 ,yerr=UDR_2_err ,width=1 ,color='grey' ,hatch='..' ,alpha=0.7 ,label="UDR-2" )
# ax3.bar( 11 ,UDR_3 ,yerr=UDR_3_err ,width=1 ,color='grey' ,hatch='--' ,alpha=0.7 ,label="UDR-3" )

ax3.set_xticks( []  )
ax3.spines['right'].set_visible( False )
ax3.spines['top'].set_visible( False )
ax3.set_title( "Cellular" )


# test-on-wild 1
RobustMPC = [2.624]
FastMPC = [1.98]
BBA=[2.57]
ADR = [3.208]
UDR_1 = [2.567]
UDR_2 = [2.748]
UDR_3 = [2.424]


RobustMPC_err = [0.41]
FastMPC_err = [0.43]
BBA_err=[0.57]
ADR_err = [0.64]
UDR_1_err = [0.77]
UDR_2_err = [0.79]
UDR_3_err = [0.58]

ax4.bar( x_loc[2] ,ADR ,yerr=ADR_err ,width=1 ,color='C2' ,label="GENET" )
ax4.bar( x_loc[1] ,RobustMPC ,yerr=RobustMPC_err ,width=1 ,color='C0' ,hatch='xx' ,alpha=1 ,label="RobustMPC" )
# ax1.bar( 5 ,FastMPC ,yerr=FastMPC_err ,width=1 ,color='C0' ,hatch='..' ,alpha=1 ,label="MPC" )
ax4.bar( x_loc[0] ,BBA ,yerr=BBA_err ,width=1 ,color='C0' ,hatch='--' ,alpha=1 ,label="BBA" )
# ax4.bar( 9 ,UDR_1 ,yerr=UDR_1_err ,width=1 ,color='grey' ,hatch='xx' ,alpha=0.7 ,label="UDR-1" )
# ax4.bar( 10 ,UDR_2 ,yerr=UDR_2_err ,width=1 ,color='grey' ,hatch='..' ,alpha=0.7 ,label="UDR-2" )
# ax4.bar( 11 ,UDR_3 ,yerr=UDR_3_err ,width=1 ,color='grey' ,hatch='--' ,alpha=0.7 ,label="UDR-3" )

ax4.set_xticks( [] )
ax4.spines['right'].set_visible( False )
ax4.spines['top'].set_visible( False )
ax4.set_title( "WiFi 2" )

# ### test-on-wild 2
RobustMPC = [1.99]
FastMPC = [1.87]
BBA = [1.587]
ADR = [2.308]
UDR_1 = [1.87]
UDR_2 = [1.648]
UDR_3 = [1.624]

RobustMPC_err = [0.42]
FastMPC_err = [0.41]
BBA_err = [0.13]
ADR_err = [0.38]
UDR_1_err = [0.28]
UDR_2_err = [0.31]
UDR_3_err = [0.37]

ax5.bar( x_loc[2] ,ADR ,yerr=ADR_err ,width=1 ,color='C2' ,label="GENET" )
ax5.bar( x_loc[1] ,RobustMPC ,yerr=RobustMPC_err ,width=1 ,color='C0' ,hatch='xx' ,alpha=1 ,label="RobustMPC" )
# ax1.bar( 5 ,FastMPC ,yerr=FastMPC_err ,width=1 ,color='C0' ,hatch='..' ,alpha=1 ,label="MPC" )
ax5.bar( x_loc[0] ,BBA ,yerr=BBA_err ,width=1 ,color='C0' ,hatch='--' ,alpha=1 ,label="BBA" )
# ax5.bar( 9 ,UDR_1 ,yerr=UDR_1_err ,width=1 ,color='grey' ,hatch='xx' ,alpha=0.7 ,label="UDR-1" )
# ax5.bar( 10 ,UDR_2 ,yerr=UDR_2_err ,width=1 ,color='grey' ,hatch='..' ,alpha=0.7 ,label="UDR-2" )
# ax5.bar( 11 ,UDR_3 ,yerr=UDR_3_err ,width=1 ,color='grey' ,hatch='--' ,alpha=0.7 ,label="UDR-3" )

ax5.set_xticks([])
ax5.spines['right'].set_visible( False )
ax5.spines['top'].set_visible( False )
ax5.set_title( "WiFi 3" )

ax1.set_ylim(bottom=1.1)
ax2.set_ylim(bottom=1.1)
ax3.set_ylim(bottom=1.1)
ax4.set_ylim(bottom=1.1)
ax5.set_ylim(bottom=1.1)


line_labels = ["GENET" ,"MPC" ,"BBA"]
