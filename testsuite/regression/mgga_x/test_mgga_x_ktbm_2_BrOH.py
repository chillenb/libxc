
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_2_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.299823046226660e+01, -2.299827574464992e+01, -2.299859589574956e+01, -2.299785671421050e+01, -2.299825329481168e+01, -2.299825329481168e+01, -3.461707559004342e+00, -3.461754535469693e+00, -3.463429830045318e+00, -3.465009484568955e+00, -3.461724518369231e+00, -3.461724518369231e+00, -6.468018934025789e-01, -6.464827872611856e-01, -6.390385929735765e-01, -6.469233971055086e-01, -6.466878316429269e-01, -6.466878316429269e-01, -1.971521458049265e-01, -1.993524463658881e-01, -7.477991153611315e-01, -1.365578641047113e-01, -1.977619914449588e-01, -1.977619914449588e-01, -1.067228717203299e-02, -1.117609074846512e-02, -4.525466898613178e-02, -5.041828812326396e-03, -1.106145853097213e-02, -1.106145853097213e-02, -5.587148706834547e+00, -5.587255242625762e+00, -5.587189063889312e+00, -5.587270364904173e+00, -5.587163468462242e+00, -5.587163468462242e+00, -2.128265693864242e+00, -2.149955278767479e+00, -2.127604610351270e+00, -2.144914777632970e+00, -2.143611485639301e+00, -2.143611485639301e+00, -5.986344430374836e-01, -6.337853992042346e-01, -5.555833931545695e-01, -5.660341422037859e-01, -6.260564506000824e-01, -6.260564506000824e-01, -1.014196860740380e-01, -1.975541105830284e-01, -1.001457484464701e-01, -1.870797454164563e+00, -1.182173135153731e-01, -1.182173135153731e-01, -4.900921269112356e-03, -5.577973034495374e-03, -4.209853899880434e-03, -6.246629623612208e-02, -5.063674855870945e-03, -5.063674855870946e-03, -6.257807881341469e-01, -6.263829269646176e-01, -6.261808883720832e-01, -6.260119149031775e-01, -6.260951411914558e-01, -6.260951411914558e-01, -6.011237386705590e-01, -5.504395151684665e-01, -5.668124130738172e-01, -5.802534584409380e-01, -5.731379068346590e-01, -5.731379068346590e-01, -6.569325378779122e-01, -2.519549663338380e-01, -2.967461115971752e-01, -3.641830434158501e-01, -3.316800486208860e-01, -3.316800486208859e-01, -4.803982851807896e-01, -4.203219543467212e-02, -5.668316067071443e-02, -3.396002674375541e-01, -8.248231079609188e-02, -8.248231079609190e-02, -1.232578771837065e-02, -1.514081365235497e-03, -2.925691827978611e-03, -7.861506244726138e-02, -4.300238809709563e-03, -4.300238809709557e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_2_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.768536530250798e+01, -2.768547167677599e+01, -2.768590869633103e+01, -2.768416526265023e+01, -2.768542174411443e+01, -2.768542174411443e+01, -4.425952056967978e+00, -4.426036509117353e+00, -4.428694751511397e+00, -4.427506713819064e+00, -4.425999794449908e+00, -4.425999794449908e+00, -8.015331807160107e-01, -8.007504821499045e-01, -7.836002277996894e-01, -7.925840217002111e-01, -8.012514778821237e-01, -8.012514778821237e-01, -2.357761468357850e-01, -2.378738983564026e-01, -9.292340364714137e-01, -1.714610277560597e-01, -2.363624936848757e-01, -2.363624936848757e-01, -1.371785654440463e-02, -1.436629797055909e-02, -5.811262108429400e-02, -6.522628682332846e-03, -1.421818952440870e-02, -1.421818952440870e-02, -6.852701209461869e+00, -6.856643428270880e+00, -6.853090159936372e+00, -6.856156177877821e+00, -6.854741108956627e+00, -6.854741108956627e+00, -2.452545216559858e+00, -2.467774955917339e+00, -2.447175228722109e+00, -2.458865959398049e+00, -2.470567630975102e+00, -2.470567630975102e+00, -7.529282932941861e-01, -8.648681620607269e-01, -7.091756852728851e-01, -7.739681915156358e-01, -7.875283370671167e-01, -7.875283370671167e-01, -1.289150927547890e-01, -2.430945841330763e-01, -1.271947803686900e-01, -2.611826460041218e+00, -1.489748266499469e-01, -1.489748266499469e-01, -6.318565955680350e-03, -7.204268942358406e-03, -5.421387367236422e-03, -8.023241163354979e-02, -6.551428866185142e-03, -6.551428866185137e-03, -8.320985308981129e-01, -8.150002230228079e-01, -8.210559955953234e-01, -8.257949987542582e-01, -8.234196808731942e-01, -8.234196808731941e-01, -8.086478355890454e-01, -6.292062082874037e-01, -6.710129834713480e-01, -7.195412156979596e-01, -6.941813795570458e-01, -6.941813795570460e-01, -9.033701697064839e-01, -3.030517813898176e-01, -3.504224883444781e-01, -4.394550786568343e-01, -3.896430747697379e-01, -3.896430747697380e-01, -5.710818392167414e-01, -5.396186945864054e-02, -7.271169471529061e-02, -4.201738472244388e-01, -1.050384804627489e-01, -1.050384804627490e-01, -1.593639522906394e-02, -1.963949914439673e-03, -3.765607472773417e-03, -1.001212729403716e-01, -5.565779429305509e-03, -5.565779429305502e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_2_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.468366424488512e-08, -1.468358639197155e-08, -1.468314523431577e-08, -1.468442988700200e-08, -1.468362387003968e-08, -1.468362387003968e-08, -1.376056283418846e-05, -1.376253205382186e-05, -1.381981867953747e-05, -1.379437439520373e-05, -1.376084043608139e-05, -1.376084043608139e-05, -4.962004275219358e-03, -4.948734666208628e-03, -4.592618157627961e-03, -4.818583744928954e-03, -4.957759469716246e-03, -4.957759469716246e-03, -6.905322605876332e-01, -6.988630915819408e-01, -1.472776587166746e-03, -4.455502590229465e-01, -6.925501899231602e-01, -6.925501899231602e-01, -6.149243510342170e+01, -5.608929235825993e+01, -2.517755537941038e+00, -1.805486672961793e+02, -5.847612647568595e+01, -5.847612647568594e+01, -4.372535789518165e-06, -4.372525697466199e-06, -4.372561977726992e-06, -4.372551827233442e-06, -4.372498450482235e-06, -4.372498450482235e-06, -1.181252418583834e-04, -1.187466031711827e-04, -1.176780810032038e-04, -1.182821447185552e-04, -1.190730690139091e-04, -1.190730690139091e-04, -2.247138934505429e-02, -1.884086176051767e-02, -2.456736121682192e-02, -2.441213098836644e-02, -2.070315727326078e-02, -2.070315727326078e-02, -6.161591262019546e-01, -3.108541331237201e-01, -6.994605091090320e-01, -1.993757847962834e-04, -6.898711467223952e-01, -6.898711467223952e-01, -2.240567626700298e+02, -1.621279503596211e+02, -5.364475818548650e+02, -1.300513403861656e+00, -2.236299295610872e+02, -2.236299295610869e+02, -2.623539717720682e-02, -2.644198990847343e-02, -2.637760942107430e-02, -2.632080211411301e-02, -2.634996068726991e-02, -2.634996068726991e-02, -2.962391521499167e-02, -3.924080157975735e-02, -3.745326805983475e-02, -3.489870100505147e-02, -3.627616768236098e-02, -3.627616768236097e-02, -1.494791848593764e-02, -2.277186544537329e-01, -1.804244108003783e-01, -1.163323713849738e-01, -1.562479649314376e-01, -1.562479649314376e-01, -4.993094092296133e-02, -2.688126991454626e+00, -1.576709788094717e+00, -1.712639463328232e-01, -1.145078969664011e+00, -1.145078969664011e+00, -3.200934863170730e+01, -2.433060539292005e+03, -8.135988051478986e+02, -1.382277247951396e+00, -3.691866057738498e+02, -3.691866057738503e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_2_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_2_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.653544797015941e-03, 1.653543732662113e-03, 1.653550272698403e-03, 1.653568881642489e-03, 1.653544123265378e-03, 1.653544123265378e-03, 5.044951095010467e-03, 5.046187692549210e-03, 5.084355048650387e-03, 5.085499500362588e-03, 5.045165411029874e-03, 5.045165411029874e-03, 7.947357695395330e-03, 7.936930246942425e-03, 7.893603179428227e-03, 9.221569453835095e-03, 7.945382229839254e-03, 7.945382229839254e-03, 6.503361055366785e-02, 6.673875614985551e-02, 3.659942694763401e-04, 1.133698205379046e-02, 6.538366652991137e-02, 6.538366652991137e-02, 1.015769868474324e-03, 1.059512631439153e-03, 2.927385818267131e-03, 2.319250575678812e-04, 1.074150459641818e-03, 1.074150459641819e-03, 6.951559546658913e-03, 6.951030746604715e-03, 6.951604761852033e-03, 6.951187235637146e-03, 6.951157549271144e-03, 6.951157549271144e-03, 1.224624225313822e-02, 1.266060825345835e-02, 1.228195732160317e-02, 1.262455037494786e-02, 1.247491588982824e-02, 1.247491588982824e-02, 4.599443954114334e-02, 4.631605731036707e-02, 3.901053771466971e-02, 4.148661919323040e-02, 4.847432216249359e-02, 4.847432216249359e-02, 7.762207572227912e-03, 3.247203201514747e-02, 8.488368132416902e-03, 1.225417145345740e-02, 1.466936982553191e-02, 1.466936982553191e-02, 3.099359773656206e-04, 3.041399462783981e-04, 4.968311182759166e-04, 3.268067341749834e-03, 2.897452556837493e-04, 2.897452556837537e-04, 5.964814829253677e-02, 5.991406674221805e-02, 5.982232806276870e-02, 5.974880574790517e-02, 5.978541938877726e-02, 5.978541938877726e-02, 6.056393008229308e-02, 6.443017249560944e-02, 6.450806546593897e-02, 6.336847442212702e-02, 6.390903287812215e-02, 6.390903287812214e-02, 4.066905480766689e-02, 4.778730139477017e-02, 5.519317851449287e-02, 5.470620801904588e-02, 6.095168820400559e-02, 6.095168820400553e-02, 5.505118434269596e-02, 2.640195302587548e-03, 3.517588355872755e-03, 6.337434722822201e-02, 8.071901730437401e-03, 8.071901730437417e-03, 6.100595945411554e-04, 7.511928990312299e-05, 2.602460467783081e-04, 8.315098686989193e-03, 2.880763975999947e-04, 2.880763975999954e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05