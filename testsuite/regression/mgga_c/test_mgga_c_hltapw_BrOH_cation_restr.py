
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_hltapw_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_hltapw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.624618589385081e-01, -1.624618494939190e-01, -1.624618715229411e-01, -1.624620043526924e-01, -1.624619258017711e-01, -1.624619258017711e-01, -1.118583438427161e-01, -1.118576570075270e-01, -1.118400354685070e-01, -1.118497501772755e-01, -1.118457390795483e-01, -1.118457390795483e-01, -6.975853657499843e-02, -6.974691282658409e-02, -6.945238355469452e-02, -6.948627886804293e-02, -6.948194154705624e-02, -6.948194154705624e-02, -4.086775662671549e-02, -4.105436430292177e-02, -7.495315761007731e-02, -3.743017378199096e-02, -3.882867779573123e-02, -3.882867779573123e-02, -7.503903160372049e-03, -7.734918488821354e-03, -2.080263209539868e-02, -5.349024276957898e-03, -6.119115706145558e-03, -6.119115706145558e-03, -1.192819551086633e-01, -1.192638878084370e-01, -1.192808771409837e-01, -1.192649421422568e-01, -1.192729395730423e-01, -1.192729395730423e-01, -9.631493862471741e-02, -9.639967036989361e-02, -9.625873517349949e-02, -9.633246054239826e-02, -9.638749016468545e-02, -9.638749016468545e-02, -6.352905339969148e-02, -6.395813984677075e-02, -6.253811424218292e-02, -6.257686909131764e-02, -6.365780522326442e-02, -6.365780522326442e-02, -3.246945462791202e-02, -4.185176217231926e-02, -3.138716648375058e-02, -9.339786853461515e-02, -3.422300235396735e-02, -3.422300235396735e-02, -4.578262929457332e-03, -5.297025254362386e-03, -4.066109230297292e-03, -2.614815767227279e-02, -4.790646167457350e-03, -4.790646167457350e-03, -5.832798374869518e-02, -5.978340729835993e-02, -5.938240815857009e-02, -5.897465774432040e-02, -5.918920414461232e-02, -5.918920414461232e-02, -5.799375148834201e-02, -5.993358816298536e-02, -6.010399689261056e-02, -5.994263299961487e-02, -6.007992039658703e-02, -6.007992039658703e-02, -6.555641566846132e-02, -4.567161564943733e-02, -4.868389288855311e-02, -5.324572527119392e-02, -5.083999155043546e-02, -5.083999155043546e-02, -5.857103442431786e-02, -2.034152350207062e-02, -2.376722282785110e-02, -5.178791824690684e-02, -2.891098622621684e-02, -2.891098622621684e-02, -9.536480276842970e-03, -2.135312613780814e-03, -3.415692466045049e-03, -2.804954387354539e-02, -4.483986140537653e-03, -4.483986140537658e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_hltapw_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_hltapw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.675243275374155e-01, -1.675243180638825e-01, -1.675243401604146e-01, -1.675244733972348e-01, -1.675243946055872e-01, -1.675243946055872e-01, -1.166216527825373e-01, -1.166209591088840e-01, -1.166031620861437e-01, -1.166129735469973e-01, -1.166089225038768e-01, -1.166089225038768e-01, -7.385401558148356e-02, -7.384212249524313e-02, -7.354074978969162e-02, -7.357543450955897e-02, -7.357099620111290e-02, -7.357099620111290e-02, -4.408691791064356e-02, -4.428088572567676e-02, -7.916349516848954e-02, -4.050792885076959e-02, -4.196534382298343e-02, -4.196534382298343e-02, -8.480947452329822e-03, -8.737604916076964e-03, -2.296895209953827e-02, -6.076451445408460e-03, -6.937982819961722e-03, -6.937982819961722e-03, -1.241134478301214e-01, -1.240952278521932e-01, -1.241123607548067e-01, -1.240962910973541e-01, -1.241043561164338e-01, -1.241043561164338e-01, -1.008945192415695e-01, -1.009804245379085e-01, -1.008375362274594e-01, -1.009122840097708e-01, -1.009680757609277e-01, -1.009680757609277e-01, -6.747189897341219e-02, -6.791204542637275e-02, -6.645509793128636e-02, -6.649487266128182e-02, -6.760397820340545e-02, -6.760397820340545e-02, -3.532049951952861e-02, -4.510939378161797e-02, -3.418455161819316e-02, -9.793577372903188e-02, -3.715761470198930e-02, -3.715761470198930e-02, -5.211506997660510e-03, -6.018184043060190e-03, -4.635208834348796e-03, -2.865985613935912e-02, -5.450118029916816e-03, -5.450118029916816e-03, -6.212996410997861e-02, -6.362610187579101e-02, -6.321398803248053e-02, -6.279485584828393e-02, -6.301540108866970e-02, -6.301540108866970e-02, -6.178623516268122e-02, -6.378042543104480e-02, -6.395552171863843e-02, -6.378971942059726e-02, -6.393078377689275e-02, -6.393078377689275e-02, -6.955079245044278e-02, -4.907116348513572e-02, -5.218790948285029e-02, -5.689710750864275e-02, -5.441519539067573e-02, -5.441519539067573e-02, -6.237988598809378e-02, -2.247504435785994e-02, -2.613263013090706e-02, -5.539353523406729e-02, -3.157903457415886e-02, -3.157903457415886e-02, -1.073222699277537e-02, -2.449905578917206e-03, -3.901411848882248e-03, -3.067028295743898e-02, -5.105518754687953e-03, -5.105518754687957e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_hltapw_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_hltapw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_hltapw_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_hltapw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_hltapw_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_hltapw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.968285178814908e-06, -2.968360617458249e-06, -2.968654010088698e-06, -2.967538786979987e-06, -2.968128918673964e-06, -2.968128918673964e-06, -3.581168781063423e-05, -3.582086226260358e-05, -3.606253745395464e-05, -3.595830342267022e-05, -3.597671548027753e-05, -3.597671548027753e-05, -3.908852129378262e-04, -3.892623561635025e-04, -3.532580708598351e-04, -3.723022483254565e-04, -3.677603118371031e-04, -3.677603118371031e-04, -2.942905708930095e-03, -3.020522610358346e-03, -1.708252779418056e-04, -1.915207643827616e-03, -2.330595679995758e-03, -2.330595679995758e-03, -7.967932861041095e-04, -8.347897746847501e-04, -1.264704529089949e-03, -5.078880505654887e-04, -6.746818419990238e-04, -6.746818419990238e-04, -6.959888584358885e-05, -7.013472919858550e-05, -6.962973322947808e-05, -7.010241192325426e-05, -6.986635965878364e-05, -6.986635965878364e-05, -8.824881566253057e-05, -9.068592851274472e-05, -8.631218365580425e-05, -8.847763803871027e-05, -9.059192860288732e-05, -9.059192860288732e-05, -1.706893588122508e-03, -2.770302641010845e-03, -1.387171686881840e-03, -1.860865155293471e-03, -1.838714200945725e-03, -1.838714200945725e-03, -1.556455446063601e-03, -1.816652357070339e-03, -1.551914697119918e-03, -2.176723958963493e-04, -2.098487873095786e-03, -2.098487873095787e-03, -3.677929625554083e-04, -4.733948869298289e-04, -1.062416620202723e-03, -1.562826936386145e-03, -7.626435027992389e-04, -7.626435027992391e-04, -1.604166493775555e-02, -8.208860595103567e-03, -9.901800742166773e-03, -1.194972512400722e-02, -1.082734663650695e-02, -1.082734663650695e-02, -1.487292263324882e-02, -1.908172122028351e-03, -2.565696357688033e-03, -3.898309042402362e-03, -3.081565149670803e-03, -3.081565149670803e-03, -2.089135714119631e-03, -1.818357656212400e-03, -1.894091327112266e-03, -2.184509153943893e-03, -2.122359069597075e-03, -2.122359069597074e-03, -1.764724887628838e-03, -1.254189941605528e-03, -1.311630191545532e-03, -3.143042287227406e-03, -1.938972734598525e-03, -1.938972734598525e-03, -6.742500499476504e-04, -2.284754661103759e-04, -5.973658815271042e-04, -1.981873497338864e-03, -8.580881557727196e-04, -8.580881557727184e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05