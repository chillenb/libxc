
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_bloc_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.241684892819172e+01, -2.241690981379935e+01, -2.241724300674280e+01, -2.241633809040571e+01, -2.241680332191209e+01, -2.241680332191209e+01, -3.413576988167301e+00, -3.413588954673394e+00, -3.414197153057165e+00, -3.416475769854247e+00, -3.414943922610729e+00, -3.414943922610729e+00, -6.595810103839879e-01, -6.593814294414758e-01, -6.576083750201214e-01, -6.646293644337922e-01, -6.621341856268528e-01, -6.621341856268528e-01, -2.053576111015341e-01, -2.063373698633178e-01, -7.590740442701177e-01, -1.770188837444192e-01, -1.874057511430262e-01, -1.874057511430262e-01, -1.008744718337829e-02, -1.061655053475223e-02, -5.766748608044980e-02, -5.828257851466640e-03, -7.322537040001222e-03, -7.322537040001222e-03, -5.477670623382255e+00, -5.478668602285725e+00, -5.477763952536852e+00, -5.478643572920938e+00, -5.478151870363083e+00, -5.478151870363083e+00, -2.092417635946117e+00, -2.106067992189892e+00, -2.086873153795455e+00, -2.098579314950179e+00, -2.102736576767984e+00, -2.102736576767984e+00, -5.957470211687707e-01, -6.094074444134405e-01, -5.388147692651737e-01, -5.336345998640641e-01, -6.039323002716199e-01, -6.039323002716199e-01, -1.380042358375376e-01, -2.226716843439643e-01, -1.291452746810737e-01, -1.808022863435683e+00, -1.524182575268732e-01, -1.524182575268732e-01, -4.497166770796330e-03, -5.697596772410557e-03, -4.356246060814597e-03, -9.069939401398992e-02, -5.247941421632613e-03, -5.247941421632613e-03, -5.830452194291993e-01, -6.015803929266536e-01, -5.978459851944697e-01, -5.929022118008668e-01, -5.956289103026263e-01, -5.956289103026263e-01, -5.460126393399084e-01, -5.206100998884901e-01, -5.418919343536486e-01, -5.604520237064492e-01, -5.514539369245010e-01, -5.514539369245010e-01, -6.361636535090169e-01, -2.631166249309992e-01, -2.970081457055513e-01, -3.621935861958890e-01, -3.269449060485891e-01, -3.269449060485891e-01, -4.760753006867702e-01, -5.530979470063142e-02, -7.446243115179481e-02, -3.444984945957508e-01, -1.113089265976243e-01, -1.113089265976243e-01, -1.422958476928548e-02, -1.523263552909914e-03, -3.197186128390381e-03, -1.056124658592880e-01, -4.856041199755400e-03, -4.856041199755396e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_bloc_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.852400444046934e+01, -2.852406149493397e+01, -2.852469353539639e+01, -2.852385439836549e+01, -2.852424003066002e+01, -2.852424003066002e+01, -4.103265948049028e+00, -4.103304092392119e+00, -4.104173034435006e+00, -4.102241298639066e+00, -4.102836912233754e+00, -4.102836912233754e+00, -7.735484435105109e-01, -7.715898041953454e-01, -7.217155611364727e-01, -7.285973789229572e-01, -7.281773722234569e-01, -7.281773722234569e-01, -1.913523447415556e-01, -1.950256862994163e-01, -9.372911876401520e-01, -1.605441142534314e-01, -1.647768539663941e-01, -1.647768539663941e-01, -1.343341436003330e-02, -1.413560716229648e-02, -7.441752673589667e-02, -7.767977537315455e-03, -9.756600067381833e-03, -9.756600067381833e-03, -6.965376803440571e+00, -6.963948725227310e+00, -6.965546769900874e+00, -6.964277043898659e+00, -6.964533421747592e+00, -6.964533421747592e+00, -2.510695677114691e+00, -2.551148849504706e+00, -2.499398497358902e+00, -2.535817531172631e+00, -2.537677021609636e+00, -2.537677021609636e+00, -7.040952112551794e-01, -7.911473711630836e-01, -6.397497571160750e-01, -6.995413975512168e-01, -7.157390691228196e-01, -7.157390691228196e-01, -1.500971668133343e-01, -1.974420008244101e-01, -1.436074147149881e-01, -2.390636953549711e+00, -1.490260405054418e-01, -1.490260405054418e-01, -5.994623227657941e-03, -7.593721696487564e-03, -5.804884070033921e-03, -1.099272109648648e-01, -6.993147667449949e-03, -6.993147667449949e-03, -7.274273828947113e-01, -7.290786470238914e-01, -7.251572143164480e-01, -7.237144007039815e-01, -7.241151988279513e-01, -7.241151988279513e-01, -7.194238086114809e-01, -6.680937868852621e-01, -6.909256693775321e-01, -6.905302764432322e-01, -6.917323452024031e-01, -6.917323452024030e-01, -8.302765597473994e-01, -2.415037418118324e-01, -3.027850606963216e-01, -4.141029165237378e-01, -3.670876243684558e-01, -3.670876243684558e-01, -5.704014942364978e-01, -7.167415183101201e-02, -9.407264835046504e-02, -4.015567308238668e-01, -1.253367626864198e-01, -1.253367626864199e-01, -1.893110742530842e-02, -2.030888001624809e-03, -4.262077999331816e-03, -1.213849501001318e-01, -6.471064238187055e-03, -6.471064238187049e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_bloc_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.054112576731739e-09, -8.054446104739757e-09, -8.053957515796122e-09, -8.048967233032986e-09, -8.051862924216908e-09, -8.051862924216908e-09, -1.490234731097894e-05, -1.490593810440494e-05, -1.500819321733381e-05, -1.503956173393095e-05, -1.500986594069856e-05, -1.500986594069856e-05, -5.263109128682630e-03, -5.313625152016340e-03, -6.467647209111832e-03, -6.650774449606090e-03, -6.562788117445477e-03, -6.562788117445477e-03, -5.942670914516548e-01, -5.776767431348419e-01, -1.397119567813111e-03, -1.035280781276045e+00, -9.364639040682436e-01, -9.364639040682438e-01, -2.440394154110498e+00, -2.578649144758224e+00, -1.482479331734616e+00, -2.215788990183559e+00, -2.815961812634830e+00, -2.815961812634837e+00, -3.608415002116778e-06, -3.664667668919837e-06, -3.609789990447069e-06, -3.659479192119576e-06, -3.637434054485256e-06, -3.637434054485256e-06, -3.930559994811851e-05, -3.410534871499953e-05, -3.563237553306273e-05, -3.090758561246206e-05, -3.840343099314925e-05, -3.840343099314925e-05, -2.753203643168115e-02, -2.121806267518437e-02, -3.001865150617308e-02, -1.487061103625039e-02, -2.853705591442120e-02, -2.853705591442120e-02, -1.142867464891124e+00, -4.477984981276115e-01, -1.246982937064819e+00, -1.096656905268813e-04, -1.406335217924097e+00, -1.406335217924097e+00, -2.919228983552001e+00, -2.607846474174909e+00, -1.638066505503951e+01, -1.750216786304757e+00, -7.622821878746962e+00, -7.622821878746946e+00, -1.845469086495038e-01, -9.390756018120459e-02, -1.196773801443563e-01, -1.473452623132012e-01, -1.327782868570078e-01, -1.327782868570078e-01, -9.518110186505706e-02, -4.977050878489161e-03, -1.317372046014696e-02, -3.949083937606736e-02, -2.349761215951772e-02, -2.349761215951772e-02, -1.333289400736407e-02, -2.230483762789774e-01, -1.244089708122505e-01, -1.056727315154631e-01, -8.954917627276694e-02, -8.954917627276700e-02, -2.985859558225906e-02, -1.347452988361871e+00, -1.370419987905233e+00, -1.690756136159394e-01, -2.016923743482696e+00, -2.016923743482694e+00, -1.968615653073977e+00, -1.279347280393323e+01, -6.192467561152518e+00, -2.018569411754196e+00, -9.628097715221482e+00, -9.628097715221500e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_bloc_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_bloc_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.064778751708119e-03, 1.064839559595129e-03, 1.064908085707846e-03, 1.064002287538641e-03, 1.064505587365269e-03, 1.064505587365269e-03, 3.631218618439428e-03, 3.632653719118424e-03, 3.675241208121760e-03, 3.702134831060520e-03, 3.680995748357480e-03, 3.680995748357480e-03, 3.976782300757182e-03, 4.030829528949791e-03, 5.230063817526429e-03, 6.405257724193986e-03, 5.978340991771586e-03, 5.978340991771586e-03, -2.123196269585137e-03, -1.596718244657967e-03, 1.788273496392971e-05, 5.158208119870122e-04, 8.424866209825733e-04, 8.424866209825712e-04, -1.598400489061846e-12, -2.582428889175856e-12, -4.363684281506855e-08, -9.192882316903506e-15, -1.452638701758989e-13, -1.452638701758976e-13, 7.978475713819516e-03, 8.108668213273208e-03, 7.983913259222054e-03, 8.098848313063400e-03, 8.044466313401170e-03, 8.044466313401170e-03, 1.156803548293948e-03, 8.159539036256723e-04, 6.135411373942149e-04, 2.845817385906669e-04, 1.255141289762305e-03, 1.255141289762305e-03, 4.854011915179338e-02, 2.960961498235219e-02, 3.264045150988020e-02, 1.012249109591956e-02, 5.289618146996308e-02, 5.289618146996308e-02, 9.385522075915018e-06, -4.515060878166160e-04, 7.980764235276438e-06, 2.724978958392418e-03, -9.067000401236449e-06, -9.067000401236551e-06, 6.519439062303287e-15, 1.512153763622152e-14, -4.306283649302779e-14, 3.908041477639475e-07, 5.227240446470207e-14, 5.227240446470113e-14, 3.283637332079594e-01, 2.208714899305778e-01, 2.676493170379120e-01, 3.078467046054419e-01, 2.881667704518600e-01, 2.881667704518599e-01, 7.282106020959650e-02, 1.394001643314219e-03, 2.072595099099462e-02, 7.349966326148437e-02, 4.151732265216298e-02, 4.151732265216297e-02, 1.970766027325397e-02, -1.894925834262295e-03, 6.462897546672850e-04, 3.412940453846620e-02, 1.043346311451832e-02, 1.043346311451831e-02, 2.469532591693655e-02, -5.562551873797557e-08, -1.141444216272219e-07, 5.367297118339875e-02, -2.445693733251546e-06, -2.445693733251564e-06, 5.813016079053462e-12, 2.071107188691128e-17, -2.285679688140848e-15, -8.783652592315253e-06, 1.292649419248049e-15, 1.292649419248426e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05