
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_cf22d_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.663694968526783e+00, -2.663336829560036e+00, -2.662011596746242e+00, -2.667315981864545e+00, -2.664499590784658e+00, -2.664499590784658e+00, -2.281997148526852e+00, -2.281494752816823e+00, -2.268299654011883e+00, -2.273660532068343e+00, -2.272906337466912e+00, -2.272906337466912e+00, -5.281461905897409e-01, -5.293042046118186e-01, -5.596623339286853e-01, -5.573235390638706e-01, -5.591335928643768e-01, -5.591335928643768e-01, -1.476876252415362e-01, -1.489339450328490e-01, -6.915840895269381e-01, -8.755460092216130e-02, -1.333676989068835e-01, -1.333676989068834e-01, 1.746559745413371e-04, 1.911490362857846e-04, 5.330423318728136e-04, 6.011161768198837e-05, 1.185221746797418e-04, 1.185221746797355e-04, -6.614410175245958e-02, -3.684556841030574e-02, -6.463018683205553e-02, -3.877936713412009e-02, -5.143377775459254e-02, -5.143377775459254e-02, -1.500655044254135e+00, -1.459709647501818e+00, -1.538538434396197e+00, -1.500185553886423e+00, -1.460177944899275e+00, -1.460177944899275e+00, -2.997030185476057e-01, -2.357124854092432e-01, -3.110486744421443e-01, -2.745256970596630e-01, -2.926984182303745e-01, -2.926984182303743e-01, -3.524132094125004e-02, -1.447323551614496e-01, -2.854962023241553e-02, -7.579388864395583e-01, -6.359912106249888e-02, -6.359912106249890e-02, 3.479787432891873e-05, 5.751868239195481e-05, 2.829360246363550e-05, -7.925246352097973e-03, 4.939811913227520e-05, 4.939811913227608e-05, -5.257766564430835e-02, -1.286631628758172e-01, -1.055282790277506e-01, -8.354755935405651e-02, -9.488619997413719e-02, -9.488619997413725e-02, -4.791608887392829e-02, -2.833337098256101e-01, -2.508582797806800e-01, -2.123435155393175e-01, -2.340041729832261e-01, -2.340041729832261e-01, -2.732763776018549e-01, -1.973363535184111e-01, -2.264492923986989e-01, -2.342119372669531e-01, -2.336952434485521e-01, -2.336952434485522e-01, -2.832014912104951e-01, 7.333897190122039e-04, -1.680241871348419e-03, -2.004436735383090e-01, -2.316803733007670e-02, -2.316803733007661e-02, 3.446956728401560e-04, 1.637478331574696e-06, 1.503661609926840e-05, -1.988857670092625e-02, 4.092482805405100e-05, 4.092482805406171e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_cf22d_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.980955815638377e+01, 1.981821428563575e+01, 1.981131780007777e+01, 1.981949110690184e+01, 1.981592645096358e+01, 1.982638290155201e+01, 1.979616733896006e+01, 1.980115772943142e+01, 1.981069200450426e+01, 1.981048037738691e+01, 1.981069200450426e+01, 1.981048037738691e+01, 1.123828800609198e-01, 1.118518462324824e-01, 1.123995604381402e-01, 1.118132966995839e-01, 1.121261296691392e-01, 1.119468917135895e-01, 1.145699005023715e-01, 1.140871510870083e-01, 1.122662172998679e-01, 1.138123717889429e-01, 1.122662172998679e-01, 1.138123717889429e-01, -3.122449171054684e-01, -3.125855922565056e-01, -3.117334218596814e-01, -3.127381705236573e-01, -3.104836047106319e-01, -3.047344317317820e-01, -2.916056833043729e-01, -2.895775884704049e-01, -3.204982905736850e-01, -2.735496991688278e-01, -3.204982905736850e-01, -2.735496991688278e-01, -1.345681495252372e-01, -1.280697173604955e-01, -1.318785047710832e-01, -1.243823509674673e-01, -5.485244737162684e-01, -5.427454690949345e-01, -1.366150057120726e-01, -1.408821136475427e-01, -1.385398062296650e-01, -8.718610219350201e-02, -1.385398062296654e-01, -8.718610219350188e-02, 2.320897175954559e-04, 2.550596909896850e-04, 2.476414963961311e-04, 2.739094802533449e-04, -4.308447294249004e-03, -5.847832035370221e-03, 9.722469483788374e-05, 9.407024467342701e-05, 1.940395973115030e-04, 6.395540113471395e-05, 1.940395973114099e-04, 6.395540113488260e-05, 7.418479038174721e+00, 7.415896897445148e+00, 7.521317936177564e+00, 7.515260403785285e+00, 7.425295489753832e+00, 7.420110538144709e+00, 7.513484816002379e+00, 7.510325243466489e+00, 7.470233641469098e+00, 7.465591520214026e+00, 7.470233641469098e+00, 7.465591520214026e+00, 2.841219795841552e-01, 2.833552330564684e-01, 2.573710200567112e-01, 2.566682026433489e-01, 3.471756128841668e-01, 3.279393102207187e-01, 3.258944845204042e-01, 3.057396396873282e-01, 2.237055793725095e-01, 2.631408713866877e-01, 2.237055793725095e-01, 2.631408713866877e-01, -1.974870859317177e-01, -1.956611931792960e-01, -6.052709924224842e-02, -4.732688800422569e-02, -2.050757912400308e-01, -1.997400541552904e-01, -2.199078836338396e-01, -1.959151340202414e-01, -1.920784710236549e-01, -1.939116723540185e-01, -1.920784710236549e-01, -1.939116723540182e-01, -8.598527098577403e-02, -8.723515465833026e-02, -1.693299042930550e-01, -1.694916937264831e-01, -6.990041911539929e-02, -7.956129502793871e-02, 1.279272923452733e-01, 1.286774725647492e-01, -1.131870689803800e-01, -1.228658139058234e-01, -1.131870689803803e-01, -1.228658139058229e-01, 5.521886301020807e-05, 6.018884388573101e-05, 9.022906564221484e-05, 9.385804678513960e-05, 3.928370497850042e-05, 4.261381650875522e-05, -2.984988617319576e-02, -3.234362103173773e-02, 5.037443514471744e-05, 8.091962890528538e-05, 5.037443514466454e-05, 8.091962890537208e-05, 1.731976501028087e-01, 1.791870152216045e-01, 9.241936555735950e-02, 9.881913410868984e-02, 1.238282800839522e-01, 1.303790007087731e-01, 1.480956662581082e-01, 1.541328688009078e-01, 1.362625984005497e-01, 1.425499052855187e-01, 1.362625984005497e-01, 1.425499052855183e-01, 1.369331258461481e-01, 1.414054659718068e-01, -1.335386233910549e-01, -1.351529000916456e-01, -1.575065370486838e-01, -1.582840523458590e-01, -1.112589885405054e-01, -1.099698355324484e-01, -1.487287093366537e-01, -1.485750019661433e-01, -1.487287093366537e-01, -1.485750019661434e-01, -1.248590625223673e-01, -1.030659092943393e-01, -1.684764831875861e-01, -1.679251756139286e-01, -1.495077031177978e-01, -1.468642836802096e-01, -1.316967067782814e-01, -1.327589779815760e-01, -1.267202413749799e-01, -1.261785927079027e-01, -1.267202413749802e-01, -1.261785927079028e-01, -1.361036455574323e-01, -1.349009034049841e-01, -4.140634143963906e-03, -4.187312685722216e-03, -1.265270222305479e-02, -1.464277888560057e-02, -1.230099373419978e-01, -1.017687288558474e-01, -5.745651395694375e-02, -6.901478059034559e-02, -5.745651395694592e-02, -6.901478059034646e-02, 4.573375872253640e-04, 4.822453326488449e-04, 3.831124464778862e-06, 3.818037596719945e-06, 2.345960910840160e-05, 2.669190745088516e-05, -5.488841497676438e-02, -5.954580742101363e-02, 4.075100657571137e-05, 6.903033411871475e-05, 4.075100657568689e-05, 6.903033411868841e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_cf22d_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.512244941551034e-08, 0.000000000000000e+00, -4.515288522195467e-08, -4.512653863492081e-08, 0.000000000000000e+00, -4.515585077095148e-08, -4.513752814611151e-08, 0.000000000000000e+00, -4.517218676013793e-08, -4.509169935424056e-08, 0.000000000000000e+00, -4.511365576311703e-08, -4.512510844192281e-08, 0.000000000000000e+00, -4.513548498197769e-08, -4.512510844192281e-08, 0.000000000000000e+00, -4.513548498197769e-08, 2.377846034823191e-06, 0.000000000000000e+00, 2.370775778243232e-06, 2.384195159012752e-06, 0.000000000000000e+00, 2.374729075423550e-06, 2.503909721088691e-06, 0.000000000000000e+00, 2.524453364500172e-06, 2.469215426925444e-06, 0.000000000000000e+00, 2.467459595414622e-06, 2.364228753245630e-06, 0.000000000000000e+00, 2.573881665453444e-06, 2.364228753245630e-06, 0.000000000000000e+00, 2.573881665453444e-06, -1.953583101833982e-02, 0.000000000000000e+00, -1.928517665965708e-02, -1.962174161749570e-02, 0.000000000000000e+00, -1.933985886383617e-02, -2.086743410010233e-02, 0.000000000000000e+00, -2.106100856989161e-02, -1.989219830514759e-02, 0.000000000000000e+00, -1.970932992456190e-02, -1.903239118092996e-02, 0.000000000000000e+00, -2.138777165038685e-02, -1.903239118092996e-02, 0.000000000000000e+00, -2.138777165038685e-02, -2.585123366352152e+00, 0.000000000000000e+00, -2.550499016394407e+00, -2.591794393542457e+00, 0.000000000000000e+00, -2.518972411844537e+00, -1.457001713708621e-02, 0.000000000000000e+00, -1.293977397827871e-02, -2.048807662908514e+00, 0.000000000000000e+00, -2.403820359274061e+00, -2.217046423717450e+00, 0.000000000000000e+00, -1.130984258722398e+00, -2.217046423717446e+00, 0.000000000000000e+00, -1.130984258722455e+00, 3.201858838563484e+00, 0.000000000000000e+00, 3.133696871392355e+00, 3.345029437191373e+00, 0.000000000000000e+00, 3.281871800797177e+00, 2.303786422711094e-01, 0.000000000000000e+00, 9.362527568523669e-02, 3.139954084560602e+00, 0.000000000000000e+00, 3.059022470308237e+00, 3.259441524538532e+00, 0.000000000000000e+00, 8.812073390814252e+00, 3.259441524519648e+00, 0.000000000000000e+00, 8.812073390385175e+00, -2.648493740507117e-05, 0.000000000000000e+00, -2.651081681545940e-05, -2.684427940622150e-05, 0.000000000000000e+00, -2.685841591227140e-05, -2.650966361108645e-05, 0.000000000000000e+00, -2.652621753920538e-05, -2.681776844275143e-05, 0.000000000000000e+00, -2.684176557205853e-05, -2.666491756583011e-05, 0.000000000000000e+00, -2.668440221530004e-05, -2.666491756583011e-05, 0.000000000000000e+00, -2.668440221530004e-05, -6.811633988104277e-05, 0.000000000000000e+00, -6.697771747489759e-05, -5.444029018781103e-05, 0.000000000000000e+00, -5.377041368681107e-05, -8.199237842501693e-05, 0.000000000000000e+00, -7.676912005907712e-05, -6.899517614096831e-05, 0.000000000000000e+00, -6.396326383823347e-05, -5.100866686590684e-05, 0.000000000000000e+00, -5.846713665217343e-05, -5.100866686590684e-05, 0.000000000000000e+00, -5.846713665217343e-05, -1.161741327942607e-02, 0.000000000000000e+00, -1.152573001932726e-02, -2.583031171107440e-02, 0.000000000000000e+00, -2.805540202309046e-02, -3.265450883501439e-02, 0.000000000000000e+00, -2.142287887318486e-02, -2.444995587485504e-02, 0.000000000000000e+00, -1.892173194087609e-02, -1.015306202282817e-02, 0.000000000000000e+00, -1.254077615437020e-02, -1.015306202282817e-02, 0.000000000000000e+00, -1.254077615437018e-02, -1.653602957897381e+00, 0.000000000000000e+00, -1.651751489153330e+00, -1.512447701059770e+00, 0.000000000000000e+00, -1.495811043054474e+00, -1.478494274687788e+00, 0.000000000000000e+00, -1.677309985459716e+00, -1.610500370364874e-04, 0.000000000000000e+00, -1.618656422875337e-04, -2.384003255006001e+00, 0.000000000000000e+00, -3.333966279565566e+00, -2.384003255006029e+00, 0.000000000000000e+00, -3.333966279565564e+00, 4.516318703483329e+00, 0.000000000000000e+00, 3.898469944468085e+00, 3.812310391922793e+00, 0.000000000000000e+00, 3.515862498591537e+00, 2.215451237064411e+01, 0.000000000000000e+00, 2.454775521206469e+01, -8.738496341491373e-01, 0.000000000000000e+00, -1.135117480125921e+00, 1.104405958240343e+01, 0.000000000000000e+00, 1.052084917877377e+01, 1.104405958306256e+01, 0.000000000000000e+00, 1.052084917886686e+01, -2.518864223283173e-01, 0.000000000000000e+00, -2.527554406421215e-01, -1.475445969842206e-01, 0.000000000000000e+00, -1.492057207864992e-01, -1.780383113758717e-01, 0.000000000000000e+00, -1.797780689869234e-01, -2.082906993275881e-01, 0.000000000000000e+00, -2.093131600793251e-01, -1.925609106730470e-01, 0.000000000000000e+00, -1.939677920386027e-01, -1.925609106730470e-01, 0.000000000000000e+00, -1.939677920386026e-01, -2.569170823822619e-01, 0.000000000000000e+00, -2.588509006097096e-01, -2.365343142148398e-02, 0.000000000000000e+00, -2.266237564045412e-02, -1.347853382589368e-02, 0.000000000000000e+00, -1.340182128593330e-02, -3.739188392177455e-02, 0.000000000000000e+00, -3.791599888824070e-02, -1.865478581532813e-02, 0.000000000000000e+00, -1.897131421007770e-02, -1.865478581532814e-02, 0.000000000000000e+00, -1.897131421007771e-02, -1.488319065224350e-02, 0.000000000000000e+00, -1.677546044506569e-02, -9.722627338520622e-01, 0.000000000000000e+00, -9.613064819641658e-01, -6.286563476111809e-01, 0.000000000000000e+00, -6.144225484177253e-01, -2.340155111617759e-01, 0.000000000000000e+00, -2.290556980145294e-01, -4.030143827963247e-01, 0.000000000000000e+00, -4.040488720427508e-01, -4.030143827963250e-01, 0.000000000000000e+00, -4.040488720427506e-01, -5.605819068134059e-02, 0.000000000000000e+00, -5.124446941595619e-02, 1.682513991922208e-01, 0.000000000000000e+00, 1.676902997662230e-01, -3.196951526762900e-01, 0.000000000000000e+00, -4.120022677277855e-01, -2.633660734866838e-01, 0.000000000000000e+00, -1.907346238512203e-01, -1.889019274356097e+00, 0.000000000000000e+00, -2.554545866456092e+00, -1.889019274355959e+00, 0.000000000000000e+00, -2.554545866456120e+00, 2.373623198343322e+00, 0.000000000000000e+00, 2.402860334821513e+00, 1.448532509063601e+01, 0.000000000000000e+00, 2.566544183009932e+01, 8.730057532383842e+00, 0.000000000000000e+00, 9.265863225826479e+00, -2.071165455752929e+00, 0.000000000000000e+00, -2.459344454262444e+00, 2.284045366769866e+01, 0.000000000000000e+00, 1.101208697388404e+01, 2.284045366752308e+01, 0.000000000000000e+00, 1.101208697355751e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_cf22d_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_cf22d_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-8.333202108717073e-03, -8.335181927261707e-03, -8.333770110076973e-03, -8.335594254408097e-03, -8.335065392642931e-03, -8.337597769819398e-03, -8.328685115855158e-03, -8.329451306673974e-03, -8.333574612040033e-03, -8.332320297514880e-03, -8.333574612040033e-03, -8.332320297514880e-03, -2.455991081042700e-02, -2.455361873111368e-02, -2.456249606627135e-02, -2.455489559489770e-02, -2.460738200784392e-02, -2.461395600854918e-02, -2.460949146701879e-02, -2.460503526405865e-02, -2.455724068634399e-02, -2.464467852244042e-02, -2.455724068634399e-02, -2.464467852244042e-02, -1.606965900558232e-02, -1.671460916832294e-02, -1.589617894573078e-02, -1.660388272847809e-02, -1.313349713492850e-02, -1.281701304757658e-02, -1.512797898281403e-02, -1.572717071372788e-02, -1.680663254072670e-02, -1.177472500245888e-02, -1.680663254072670e-02, -1.177472500245888e-02, 9.026279602670208e-02, 9.285579245966424e-02, 8.905353035882946e-02, 9.015051027253586e-02, -3.304972103798272e-03, -5.788170752052685e-03, 5.438883875435861e-02, 6.686002305833777e-02, 8.413878621608170e-02, 2.668383364407876e-02, 8.413878621608180e-02, 2.668383364407828e-02, 3.555690963453984e-05, 4.286314604588828e-05, 4.244089975806108e-05, 5.288626974097103e-05, 3.337344264926463e-03, 4.086723745085170e-03, 5.624287775508529e-06, 5.401883215427161e-06, 2.394812055335283e-05, 7.509740761131882e-06, 2.394812055335282e-05, 7.509740761100954e-06, -5.701034801576454e-02, -5.700760164806080e-02, -5.804002595389365e-02, -5.800259662451555e-02, -5.707427640017874e-02, -5.704672256222755e-02, -5.795732230397440e-02, -5.795017383135841e-02, -5.753003364948155e-02, -5.750431188744091e-02, -5.753003364948155e-02, -5.750431188744091e-02, -3.901203000364386e-02, -3.911246470612659e-02, -3.943769834160613e-02, -3.950913543831774e-02, -3.908650770665589e-02, -3.917237627687865e-02, -3.957981866886505e-02, -3.963366717440048e-02, -3.912670120645769e-02, -3.932662313637417e-02, -3.912670120645769e-02, -3.932662313637417e-02, -6.817241312440261e-02, -6.963804928810122e-02, -1.454585965864486e-01, -1.533742767669352e-01, -4.307787669628504e-02, -5.513599764032105e-02, -5.679697395317526e-02, -7.180499878666079e-02, -7.508366257590268e-02, -7.066580371158959e-02, -7.508366257590264e-02, -7.066580371158958e-02, 3.300699985755238e-02, 3.323208026821152e-02, 7.205587280091308e-02, 7.160985991701890e-02, 2.676219396526821e-02, 3.154376360683497e-02, -4.951449861813881e-02, -4.957615878210594e-02, 5.188343725574109e-02, 7.019769751863056e-02, 5.188343725574108e-02, 7.019769751863050e-02, 1.722434626799375e-06, 1.779868777747114e-06, 4.749834758390532e-06, 4.380473004476588e-06, 1.426094676514412e-05, 1.859836773292812e-05, 1.375596766077614e-02, 1.584616267629040e-02, 2.863479060183063e-06, 2.361611313156300e-05, 2.863479060170783e-06, 2.361611313157076e-05, -2.382716813955275e-01, -2.415442391093773e-01, -1.903458908568061e-01, -1.927445736923761e-01, -2.079209188078696e-01, -2.107009677153084e-01, -2.224481543636284e-01, -2.253162543792240e-01, -2.153017586395838e-01, -2.181000514252405e-01, -2.153017586395838e-01, -2.181000514252407e-01, -2.912531476385260e-01, -2.907076608109084e-01, -8.604938389668301e-02, -8.650189457668958e-02, -9.903524059747472e-02, -9.910913625320004e-02, -1.217303589049822e-01, -1.216963497906820e-01, -1.067274245647340e-01, -1.064870987402858e-01, -1.067274245647339e-01, -1.064870987402858e-01, -1.051346174252454e-01, -1.156262419671870e-01, 6.404481036564891e-02, 6.349334455388050e-02, 3.889601304675699e-02, 3.659315199216619e-02, -2.749232155151446e-02, -2.763539110064674e-02, 1.990672505246554e-03, 1.428612544402049e-03, 1.990672505246563e-03, 1.428612544401927e-03, -6.073526236134174e-02, -6.529934595602964e-02, 3.343751230406754e-03, 3.341335603785502e-03, 6.548999169527587e-03, 7.425644865572110e-03, -4.829387194150936e-02, -8.901095130712784e-02, 2.791992958615417e-02, 3.707429501538289e-02, 2.791992958615392e-02, 3.707429501538288e-02, 5.043064356025538e-05, 5.531815509018277e-05, 1.055869543653308e-07, 1.058384042695265e-07, 2.472915302180752e-06, 3.152932574479204e-06, 2.877771916482510e-02, 3.337680701787834e-02, 6.276306390278951e-06, 1.883956449395918e-05, 6.276306390284275e-06, 1.883956449395195e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05