
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_hjs_b97x_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.029768322270974e+01, -2.029772381415399e+01, -2.029794776986618e+01, -2.029734296044662e+01, -2.029765368307161e+01, -2.029765368307161e+01, -3.286873897213838e+00, -3.286855877615871e+00, -3.286521705875138e+00, -3.287739131684911e+00, -3.286933282673742e+00, -3.286933282673742e+00, -6.008602632027505e-01, -6.006295025615772e-01, -5.968783583252386e-01, -6.015720772648994e-01, -6.033920279522444e-01, -6.033920279522444e-01, -1.432597955266561e-01, -1.437478631187212e-01, -7.061026052762327e-01, -1.238258231647561e-01, -1.421207664308363e-01, -1.421207664308362e-01, -4.444899844643591e-05, -5.525530941704884e-05, -2.668084363481410e-02, -5.893642427603317e-06, -2.122166764325398e-05, -2.122166764325398e-05, -4.864721651302792e+00, -4.864857755011780e+00, -4.864734945454105e+00, -4.864854970503339e+00, -4.864786667828955e+00, -4.864786667828955e+00, -1.923953380727641e+00, -1.934391772553425e+00, -1.922640374681789e+00, -1.931782881859145e+00, -1.930472544779434e+00, -1.930472544779434e+00, -5.014144855493058e-01, -5.364668530451884e-01, -4.621532902969095e-01, -4.702233272815886e-01, -5.092949640502569e-01, -5.092949640502570e-01, -9.991751382285066e-02, -1.652080123649204e-01, -9.249124210138746e-02, -1.734335351770060e+00, -1.052180975302369e-01, -1.052180975302369e-01, -2.585457111244563e-06, -5.471352063126185e-06, -2.352685719986070e-06, -5.780541726563738e-02, -5.130681526121595e-06, -5.130681526123936e-06, -4.890509368483968e-01, -4.885375369104460e-01, -4.887104824866365e-01, -4.888556584336458e-01, -4.887817591152346e-01, -4.887817591152346e-01, -4.735042886054603e-01, -4.243773555034139e-01, -4.371026732586391e-01, -4.506384904587112e-01, -4.434966818693156e-01, -4.434966818693156e-01, -5.657138923045163e-01, -2.002366246097083e-01, -2.290300100535537e-01, -2.858713525637039e-01, -2.534330131962861e-01, -2.534330131962861e-01, -3.867496801113417e-01, -2.444036569583988e-02, -4.339947938142548e-02, -2.674547267151787e-01, -7.378816233596136e-02, -7.378816233596142e-02, -1.984159772461133e-04, -9.705935201306571e-08, -9.127816039928474e-07, -6.939374179617067e-02, -3.865427835836837e-06, -3.865427835840635e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_hjs_b97x_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.562976624929776e+01, -2.562973525413875e+01, -2.562987543202088e+01, -2.562981455644479e+01, -2.563020609848210e+01, -2.563029359283736e+01, -2.562897353401748e+01, -2.562872242871377e+01, -2.562983230894454e+01, -2.562933677075383e+01, -2.562983230894454e+01, -2.562933677075383e+01, -4.073283621065482e+00, -4.073450841861847e+00, -4.073324559690856e+00, -4.073497991584251e+00, -4.074379973234761e+00, -4.074680606487860e+00, -4.073217121429978e+00, -4.073521493731580e+00, -4.072504027666067e+00, -4.074534762820356e+00, -4.072504027666067e+00, -4.074534762820356e+00, -7.121041717749704e-01, -7.181623326502350e-01, -7.098669043060252e-01, -7.173096364894993e-01, -6.796488471642005e-01, -6.688588776908136e-01, -6.796229031770252e-01, -6.831147717040936e-01, -7.285072148523507e-01, -6.213547342556967e-01, -7.285072148523507e-01, -6.213547342556967e-01, -1.039685777813361e-01, -1.140304118235094e-01, -1.081438840940227e-01, -1.198500456783909e-01, -8.284196869755215e-01, -8.712517459595732e-01, -6.459492659334778e-02, -6.599612731991664e-02, -1.118168517479998e-01, -5.429780008722121e-02, -1.118168517479990e-01, -5.429780008722176e-02, -9.024083055641066e-05, -1.175446315271960e-04, -1.105146749742854e-04, -1.497761525567548e-04, -3.620109673575016e-02, -3.900561121880589e-02, -1.268254080888893e-05, -1.197345210372310e-05, -5.579280151128246e-05, -7.210556609497169e-06, -5.579280151127404e-05, -7.210556609497874e-06, -6.251818734478762e+00, -6.250262211756380e+00, -6.254085356190489e+00, -6.252452094824274e+00, -6.251946209678721e+00, -6.250338564887716e+00, -6.253891139760632e+00, -6.252327998265737e+00, -6.252978956912042e+00, -6.251362264014455e+00, -6.252978956912042e+00, -6.251362264014455e+00, -2.129479585988638e+00, -2.129354700081100e+00, -2.150848632406943e+00, -2.150123410689218e+00, -2.101850260527166e+00, -2.109808439262405e+00, -2.120519865516348e+00, -2.128559184679770e+00, -2.161557027692662e+00, -2.143092191827659e+00, -2.161557027692662e+00, -2.143092191827659e+00, -6.433066111838374e-01, -6.415253800514727e-01, -7.216123167455522e-01, -7.221605611937355e-01, -5.738984205850526e-01, -5.995076889663782e-01, -6.190627384637405e-01, -6.413094341247856e-01, -6.746592268265698e-01, -6.372092637001076e-01, -6.746592268265698e-01, -6.372092637001077e-01, -6.392364741141102e-02, -6.299961322573244e-02, -1.012483571298637e-01, -1.020319852305892e-01, -6.268576809489149e-02, -6.223442985104391e-02, -2.295072639123225e+00, -2.294046272394921e+00, -5.615480512644373e-02, -5.099059541686684e-02, -5.615480512644373e-02, -5.099059541686684e-02, -4.931046734475197e-06, -5.573847302595530e-06, -1.111830302859701e-05, -1.170079545173919e-05, -4.320039367403698e-06, -5.178746573685292e-06, -5.400510344389731e-02, -5.488925010361934e-02, -4.618227323230580e-06, -1.307410769010725e-05, -4.618227323233132e-06, -1.307410769009855e-05, -6.658120227181562e-01, -6.686695305098131e-01, -6.580713084989518e-01, -6.609799192226403e-01, -6.608110075626432e-01, -6.637180685347958e-01, -6.630764239327720e-01, -6.659354668985665e-01, -6.619458974604656e-01, -6.648280444630661e-01, -6.619458974604656e-01, -6.648280444630661e-01, -6.473820306182340e-01, -6.497231487598856e-01, -5.098031951560725e-01, -5.127883840240728e-01, -5.509746993073249e-01, -5.541700938385299e-01, -5.910944480932023e-01, -5.935066180717141e-01, -5.711097604889639e-01, -5.736096868287355e-01, -5.711097604889639e-01, -5.736096868287355e-01, -7.581101864379086e-01, -7.597600117510329e-01, -1.474049409765589e-01, -1.491169874152307e-01, -2.082709190154131e-01, -2.123474764831133e-01, -3.392528041325165e-01, -3.417869480275344e-01, -2.715612473387061e-01, -2.718343989180355e-01, -2.715612473387060e-01, -2.718343989180356e-01, -4.631066754574935e-01, -4.676328359263561e-01, -3.571472300383751e-02, -3.607592555409358e-02, -5.112340067526454e-02, -5.228632874249273e-02, -3.320292906634332e-01, -3.387735316809204e-01, -5.380266560958184e-02, -4.808361824341330e-02, -5.380266560958159e-02, -4.808361824341324e-02, -4.499100815983315e-04, -5.283443988234302e-04, -1.935127359172167e-07, -1.948439511409651e-07, -1.649225704256736e-06, -1.990715858190399e-06, -5.134986253523234e-02, -5.170503197273205e-02, -3.893746638493170e-06, -9.731414085044752e-06, -3.893746638519408e-06, -9.731414085083072e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_hjs_b97x_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.365168136195923e-09, 0.000000000000000e+00, -7.365205602180307e-09, -7.365067055877452e-09, 0.000000000000000e+00, -7.365132173522678e-09, -7.364732801209803e-09, 0.000000000000000e+00, -7.364656260544143e-09, -7.365874639213089e-09, 0.000000000000000e+00, -7.366111563395201e-09, -7.365108399525573e-09, 0.000000000000000e+00, -7.365521783244397e-09, -7.365108399525573e-09, 0.000000000000000e+00, -7.365521783244397e-09, -1.055520729697571e-05, 0.000000000000000e+00, -1.055548450244521e-05, -1.055503796869360e-05, 0.000000000000000e+00, -1.055537777917157e-05, -1.055155966609137e-05, 0.000000000000000e+00, -1.055040081356365e-05, -1.055077203391093e-05, 0.000000000000000e+00, -1.054985313610416e-05, -1.056238571881777e-05, 0.000000000000000e+00, -1.054595111406062e-05, -1.056238571881777e-05, 0.000000000000000e+00, -1.054595111406062e-05, -7.630144777463963e-03, 0.000000000000000e+00, -7.500809590745258e-03, -7.680747775818541e-03, 0.000000000000000e+00, -7.521613341099210e-03, -8.381090686689959e-03, 0.000000000000000e+00, -8.620006212779393e-03, -8.260257869089715e-03, 0.000000000000000e+00, -8.187708215614263e-03, -7.258795018698107e-03, 0.000000000000000e+00, -9.477722725902027e-03, -7.258795018698107e-03, 0.000000000000000e+00, -9.477722725902027e-03, -1.595127494964371e+00, 0.000000000000000e+00, -1.390898895332796e+00, -1.532571225005401e+00, 0.000000000000000e+00, -1.307011999459340e+00, -4.398798929577359e-03, 0.000000000000000e+00, -3.835039315332011e-03, -3.274508747859027e+00, 0.000000000000000e+00, -3.136171259937445e+00, -1.352583103511375e+00, 0.000000000000000e+00, -6.558610652499945e+00, -1.352583103511380e+00, 0.000000000000000e+00, -6.558610652499908e+00, -1.653518804731116e-01, 0.000000000000000e+00, -2.210121725668916e-01, -2.195432947645192e-01, 0.000000000000000e+00, -3.049499390850374e-01, -1.097963904820201e+01, 0.000000000000000e+00, -1.125588451536373e+01, -1.004047190143902e-02, 0.000000000000000e+00, -8.892034710471312e-03, -9.236213212450346e-02, 0.000000000000000e+00, -1.065219200184581e-02, -9.236213212452631e-02, 0.000000000000000e+00, -1.065219200184528e-02, -2.069283771834944e-06, 0.000000000000000e+00, -2.071282144079012e-06, -2.067366647031871e-06, 0.000000000000000e+00, -2.069428007740182e-06, -2.069170629054078e-06, 0.000000000000000e+00, -2.071213680598048e-06, -2.067525689758185e-06, 0.000000000000000e+00, -2.069529336383178e-06, -2.068306734935421e-06, 0.000000000000000e+00, -2.070351586162414e-06, -2.068306734935421e-06, 0.000000000000000e+00, -2.070351586162414e-06, -1.057742631070351e-04, 0.000000000000000e+00, -1.057955675907530e-04, -1.027404697262559e-04, 0.000000000000000e+00, -1.028517794041288e-04, -1.083512736805868e-04, 0.000000000000000e+00, -1.076292072521750e-04, -1.056455082078677e-04, 0.000000000000000e+00, -1.049096662945906e-04, -1.022121623846735e-04, 0.000000000000000e+00, -1.039822149211445e-04, -1.022121623846735e-04, 0.000000000000000e+00, -1.039822149211445e-04, -1.236584465830557e-02, 0.000000000000000e+00, -1.249285236464564e-02, -8.472470739949315e-03, 0.000000000000000e+00, -8.449788104934372e-03, -1.830209553614572e-02, 0.000000000000000e+00, -1.580571358866967e-02, -1.466121907678527e-02, 0.000000000000000e+00, -1.292550363028709e-02, -1.052055924433229e-02, 0.000000000000000e+00, -1.288532745433355e-02, -1.052055924433230e-02, 0.000000000000000e+00, -1.288532745433356e-02, -5.886803109502178e+00, 0.000000000000000e+00, -5.899464635654973e+00, -1.259227683501523e+00, 0.000000000000000e+00, -1.240410057888910e+00, -7.065163868558589e+00, 0.000000000000000e+00, -6.586327888409149e+00, -1.088836921482987e-04, 0.000000000000000e+00, -1.090748470784464e-04, -5.363014902205104e+00, 0.000000000000000e+00, -5.464762407263627e+00, -5.363014902205104e+00, 0.000000000000000e+00, -5.464762407263627e+00, -2.751518310440134e-03, 0.000000000000000e+00, -2.972258796998562e-03, -9.771218326813630e-03, 0.000000000000000e+00, -9.818093842200091e-03, -1.054638748103258e-02, 0.000000000000000e+00, -1.630662378728409e-02, -1.232631399194208e+01, 0.000000000000000e+00, -1.192176612041665e+01, -5.941151619287742e-03, 0.000000000000000e+00, -3.532626369506289e-02, -5.941151619225307e-03, 0.000000000000000e+00, -3.532626369513242e-02, -1.129163977316693e-02, 0.000000000000000e+00, -1.111907221465746e-02, -1.180049893166771e-02, 0.000000000000000e+00, -1.161590800673423e-02, -1.162452334922928e-02, 0.000000000000000e+00, -1.144304662327182e-02, -1.147607732662332e-02, 0.000000000000000e+00, -1.130025253861958e-02, -1.155054233579842e-02, 0.000000000000000e+00, -1.137193350448759e-02, -1.155054233579842e-02, 0.000000000000000e+00, -1.137193350448759e-02, -1.245202811053814e-02, 0.000000000000000e+00, -1.229268823801435e-02, -2.580529268436658e-02, 0.000000000000000e+00, -2.535573803933223e-02, -2.105892293876408e-02, 0.000000000000000e+00, -2.067456493232595e-02, -1.709684332911472e-02, 0.000000000000000e+00, -1.685628098938705e-02, -1.900544660949342e-02, 0.000000000000000e+00, -1.872326784666693e-02, -1.900544660949342e-02, 0.000000000000000e+00, -1.872326784666693e-02, -7.084203601667910e-03, 0.000000000000000e+00, -7.029745880357983e-03, -5.881085491410030e-01, 0.000000000000000e+00, -5.765306474010132e-01, -3.068994944968519e-01, 0.000000000000000e+00, -2.964090738207327e-01, -1.013254784305304e-01, 0.000000000000000e+00, -9.911601174559988e-02, -1.769868095449702e-01, 0.000000000000000e+00, -1.771595343739635e-01, -1.769868095449708e-01, 0.000000000000000e+00, -1.771595343739635e-01, -3.591494604733937e-02, 0.000000000000000e+00, -3.491960303871627e-02, -1.001783891501251e+01, 0.000000000000000e+00, -1.005927939403216e+01, -1.050978192390689e+01, 0.000000000000000e+00, -1.069598016498285e+01, -1.161272803463984e-01, 0.000000000000000e+00, -1.092859865994495e-01, -1.075454976738051e+01, 0.000000000000000e+00, -1.153220537098261e+01, -1.075454976738057e+01, 0.000000000000000e+00, -1.153220537098261e+01, -6.918609975599972e-01, 0.000000000000000e+00, -8.197087211278423e-01, -1.648443087674577e-05, 0.000000000000000e+00, -2.957797214367112e-05, -6.677050078011274e-04, 0.000000000000000e+00, -1.019419748697906e-03, -1.207848245674136e+01, 0.000000000000000e+00, -1.173985445436724e+01, -8.931596603667432e-03, 0.000000000000000e+00, -2.249827356954824e-02, -8.931596603643611e-03, 0.000000000000000e+00, -2.249827356966052e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05