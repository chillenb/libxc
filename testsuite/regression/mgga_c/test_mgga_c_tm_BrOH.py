
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tm_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.455404656367153e-02, -3.455438931572780e-02, -3.455427196563252e-02, -3.454860548913458e-02, -3.455424220510953e-02, -3.455424220510953e-02, -3.878352604311272e-02, -3.878691372413096e-02, -3.887683564980463e-02, -3.867135561106134e-02, -3.878557060570887e-02, -3.878557060570887e-02, -2.495243623882001e-02, -2.473372939720895e-02, -1.913642188887472e-02, -1.930818994323574e-02, -2.487340781833873e-02, -2.487340781833873e-02, -7.533374462040545e-03, -8.125114792604459e-03, -3.257039871929561e-02, -2.220962421145000e-03, -7.719008950164818e-03, -7.719008950164818e-03, -1.070205034844705e-07, -1.326532587595502e-07, -2.860629149144250e-05, -2.825678570074668e-09, -1.312879351023905e-07, -1.312879351023905e-07, -3.545566116850323e-02, -3.553761687678889e-02, -3.546029665634443e-02, -3.552420527556158e-02, -3.550272330172070e-02, -3.550272330172070e-02, -1.669662797476365e-02, -1.688185420203343e-02, -1.613822478351030e-02, -1.627580585036542e-02, -1.757710930366934e-02, -1.757710930366934e-02, -3.401499329144043e-02, -5.436811379418161e-02, -3.434379378259278e-02, -4.940234130662599e-02, -3.532410556893113e-02, -3.532410556893113e-02, -5.642638434376121e-04, -3.685257725773644e-03, -6.212467718403447e-04, -7.076426505910505e-02, -1.219429333074611e-03, -1.219429333074611e-03, -2.445988466961987e-09, -4.646637071195341e-09, -3.356859647711346e-09, -1.258437480091301e-04, -4.535676735957949e-09, -4.535676735957950e-09, -5.372034283195503e-02, -4.072535470129957e-02, -4.398801578229691e-02, -4.739448083409078e-02, -4.557553712165375e-02, -4.557553712165375e-02, -5.988746402882578e-02, -2.085334778115746e-02, -2.563679149840494e-02, -3.195065881604866e-02, -2.861109228432194e-02, -2.861109228432195e-02, -5.480776542677137e-02, -6.143141816815511e-03, -1.015399693740877e-02, -2.135031374188429e-02, -1.489634352608482e-02, -1.489634352608482e-02, -2.267854509888988e-02, -1.695769380211047e-05, -6.527808024194930e-05, -2.599308975733721e-02, -3.899491545282027e-04, -3.899491545282028e-04, -1.771646384603097e-07, -1.874198023711227e-11, -3.094703044023785e-10, -3.959269984497373e-04, -3.121510086276105e-09, -3.121510090654696e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tm_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.051922708217971e-01, -1.051930792686404e-01, -1.051937177848854e-01, -1.051804028090039e-01, -1.051927237408927e-01, -1.051927237408927e-01, -1.053422378241726e-01, -1.053466392591046e-01, -1.054630175869105e-01, -1.051887541713084e-01, -1.053449702276012e-01, -1.053449702276012e-01, -7.463227095694779e-02, -7.432034727076695e-02, -6.496319564832306e-02, -6.527192871625938e-02, -7.451986271512757e-02, -7.451986271512757e-02, -3.155841040175678e-02, -3.322810044213727e-02, -8.576766147210942e-02, -1.177657251449687e-02, -3.209163615256999e-02, -3.209163615256999e-02, -7.293549868695174e-07, -9.026923658122738e-07, -1.850906205415742e-04, -1.894657394323800e-08, -8.938839590734008e-07, -8.938839590734009e-07, -1.068210686133776e-01, -1.070320808858848e-01, -1.068374915267448e-01, -1.070018152121938e-01, -1.069360821827158e-01, -1.069360821827158e-01, -6.264782810912904e-02, -6.304454901510101e-02, -6.125431216190666e-02, -6.155343933162441e-02, -6.474896468728322e-02, -6.474896468728322e-02, -7.987524431673707e-02, -8.335127862667696e-02, -7.967187473951337e-02, -8.237054291127038e-02, -8.112742735507918e-02, -8.112742735507918e-02, -3.355738758523886e-03, -1.847005004090541e-02, -3.674465570035671e-03, -1.179309292495700e-01, -6.906220181078823e-03, -6.906220181078823e-03, -1.660869225911490e-08, -3.127916996939768e-08, -2.296384570595815e-08, -7.848243449362325e-04, -3.039960767401064e-08, -3.039960767090046e-08, -8.120333012140594e-02, -8.453787434164678e-02, -8.473725476245411e-02, -8.413005308034352e-02, -8.454904463596971e-02, -8.454904463596971e-02, -7.514578110875011e-02, -6.297708823035673e-02, -6.980681845157094e-02, -7.692785523028167e-02, -7.343575687803713e-02, -7.343575687803716e-02, -8.509470297717839e-02, -2.790750932732321e-02, -4.003765169904052e-02, -6.168847329370473e-02, -5.076723094782641e-02, -5.076723094782638e-02, -6.559995859449505e-02, -1.108418093152368e-04, -4.160296499684419e-04, -6.514156710563893e-02, -2.362393941252866e-03, -2.362393941252885e-03, -1.174774929219603e-06, -1.261486687868391e-10, -2.134408805138756e-09, -2.394627026203922e-03, -2.093573667493624e-08, -2.093573668034501e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tm_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.117996393903377e-10, 3.118080411207198e-10, 3.118309873313121e-10, 3.116941699212532e-10, 3.118041859355219e-10, 3.118041859355219e-10, 1.206203847101861e-06, 1.206447307169771e-06, 1.213082934923979e-06, 1.202895058908586e-06, 1.206289247163945e-06, 1.206289247163945e-06, 1.778352372862416e-03, 1.764076505586881e-03, 1.382526193131378e-03, 1.365566860159478e-03, 1.773259387000683e-03, 1.773259387000683e-03, 2.200042317867545e-01, 2.304525421204974e-01, 1.027362386620384e-03, 1.506120142378146e-01, 2.234210285554014e-01, 2.234210285554014e-01, 2.949237696880079e-02, 3.177913210923104e-02, 6.529092689573500e-02, 5.521269842030506e-03, 3.312305698918599e-02, 3.312305698918599e-02, 5.736791321105832e-07, 5.795390972391348e-07, 5.742855116634862e-07, 5.788401240436814e-07, 5.766587013830414e-07, 5.766587013830414e-07, 6.786230849474864e-06, 6.823141079785395e-06, 6.548876401928057e-06, 6.578020662868037e-06, 7.132686920668982e-06, 7.132686920668982e-06, 8.873346019279145e-03, 1.159356483439329e-02, 9.623497677342933e-03, 1.322130308998678e-02, 8.954992932702801e-03, 8.954992932702801e-03, 9.403218383793005e-02, 7.005826009817016e-02, 1.159510934162466e-01, 6.786876369338053e-05, 1.470061636406620e-01, 1.470061636406620e-01, 5.690096896944087e-03, 7.087648550805025e-03, 2.145473204706933e-02, 1.053420522199578e-01, 1.094823855912661e-02, 1.094823855932268e-02, 8.155885878682571e-02, 5.196357357444584e-02, 6.201850909943996e-02, 7.123304496068289e-02, 6.649904698592192e-02, 6.649904698592193e-02, 4.325232113687032e-02, 1.164544238888077e-02, 1.631009869265535e-02, 2.505100716483387e-02, 1.982470872793337e-02, 1.982470872793337e-02, 8.365226146858060e-03, 5.080054897853460e-02, 4.663180234547015e-02, 4.264392440278750e-02, 4.673310269441934e-02, 4.673310269441931e-02, 1.543797531737163e-02, 4.501254782166746e-02, 7.022639836090648e-02, 7.811927539811353e-02, 1.562775073669235e-01, 1.562775073669234e-01, 2.438277557811245e-02, 1.779844042154383e-03, 4.326680952694393e-03, 2.020877176298537e-01, 1.480961074966782e-02, 1.480961074917513e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tm_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tm_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.082179665466739e-05, -3.082316402837036e-05, -3.082982614286409e-05, -3.080764474430522e-05, -3.082251026468869e-05, -3.082251026468869e-05, -6.857360672837423e-05, -6.860965711492635e-05, -6.973201864903013e-05, -6.976297759031697e-05, -6.857991200793953e-05, -6.857991200793953e-05, -3.017656793150496e-04, -3.003324742316562e-04, -2.604321928984404e-04, -3.098519442349337e-04, -3.013262422675194e-04, -3.013262422675194e-04, -4.806005576414724e-03, -5.252571225203471e-03, -4.288615700453534e-05, -3.076565591655384e-04, -4.925015200122001e-03, -4.925015200122001e-03, -6.695672936559823e-08, -8.112127462792107e-08, -6.900389846069505e-06, -5.373970831360682e-10, -8.289608612633891e-08, -8.289608612633881e-08, -1.019225818624581e-03, -1.031772251253069e-03, -1.020666978310011e-03, -1.030411903633195e-03, -1.025409135804615e-03, -1.025409135804615e-03, -2.456146526835099e-04, -2.689808881905752e-04, -2.383660610849871e-04, -2.564510630963338e-04, -2.714017942781446e-04, -2.714017942781446e-04, -7.206624197476073e-03, -4.571056361246281e-03, -4.025817018320344e-03, -2.818794107350280e-03, -9.975429977588402e-03, -9.975429977588402e-03, -9.963025750836761e-05, -1.140026120332008e-03, -1.197895414619745e-04, -1.778866810184747e-04, -3.182095583435709e-04, -3.182095583435709e-04, -8.607113622581716e-10, -1.181887594435025e-09, -2.508954835806010e-09, -1.655861714485900e-05, -1.063827945592203e-09, -1.063827945592201e-09, -1.207237198272500e-01, -1.174521223697289e-01, -1.305206169427635e-01, -1.363111324624062e-01, -1.342463590369714e-01, -1.342463590369715e-01, -2.215626610073263e-02, -1.387654351578501e-02, -2.414943046335434e-02, -4.182031365589198e-02, -3.117859297999269e-02, -3.117859297999269e-02, -2.657198757106702e-03, -2.275197424106563e-03, -3.784477935682244e-03, -6.425834444533591e-03, -5.980707622095944e-03, -5.980707622095924e-03, -7.431094755081813e-03, -4.519565930424555e-06, -1.323862640599836e-05, -9.690698775994978e-03, -9.763425410291165e-05, -9.763425410291160e-05, -3.257301272684304e-08, -3.595053253218561e-12, -1.944793080820651e-10, -1.060097019194223e-04, -8.503385391572767e-10, -8.503385375258043e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05